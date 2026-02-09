"""CMA-ES optimizer with simulation workflow."""

import numpy as np
from pathlib import Path
from cmaes import SepCMA
import time
from concurrent.futures import ProcessPoolExecutor


def _evaluate_analytical_worker(args):
    """Worker function for analytical evaluation (no MD simulation).
    
    Used when evaluator.requires_simulation == False.
    Since evaluators with functions can't be pickled, we pass the evaluator directly.
    
    Args:
        args: Tuple of (evaluator, params, normalized_params, gen, ind)
    
    Returns:
        (normalized_params, score, time_elapsed)
    """
    (evaluator, params, normalized_params, gen, ind) = args
    
    start_time = time.time()
    
    try:
        # Evaluate directly with parameters (no simulation!)
        score = evaluator.evaluate(params)
        
        elapsed = time.time() - start_time
        
        print(f"  [{gen:03d}.{ind:03d}] Score: {score:.6f} ({elapsed*1000:.1f}ms)")
        
        return (ind, normalized_params, score, elapsed)
        
    except Exception as e:
        print(f"  [{gen:03d}.{ind:03d}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return (ind, normalized_params, 1e6, time.time() - start_time)


def _evaluate_worker(args):
    """Worker function for multiprocessing.
    
    Reconstructs bias/sampler/evaluator from class + kwargs.
    
    Args:
        args: Tuple of (bias_class, bias_kwargs, sampler_class, sampler_kwargs,
                       evaluator_class, evaluator_kwargs, params, normalized_params, 
                       gen, ind, output_base, n_replicas)
    
    Returns:
        (normalized_params, score, time_elapsed)
    """
    (bias_class, bias_kwargs, 
     sampler_class, sampler_kwargs,
     evaluator_class, evaluator_kwargs,
     params, normalized_params, gen, ind, output_base, n_replicas) = args
    
    start_time = time.time()
    
    try:
        # Reconstruct objects in worker process
        bias = bias_class(**bias_kwargs)
        evaluator = evaluator_class(**evaluator_kwargs)
        
        # Set parameters
        bias.set_parameters(params)
        
        # Create output directory
        output_dir = Path(output_base) / f"gen{gen:03d}" / f"ind{ind:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run N replicas and average scores (reduces stochastic noise)
        scores = []
        for replica_id in range(n_replicas):
            # Create sampler for this replica
            sampler = sampler_class(**sampler_kwargs)
            
            # Create replica subdirectory
            if n_replicas > 1:
                replica_dir = output_dir / f"replica_{replica_id}"
            else:
                replica_dir = output_dir
            replica_dir.mkdir(parents=True, exist_ok=True)
            
            # Use replica_id as seed for PDB selection
            # This ensures each replica uses a different initial structure
            # With 2 replicas and 2 PDB files: replica_0 → pp1, replica_1 → pp2
            pdb_seed = replica_id
            
            # Run simulation with deterministic seed
            sampler.run(str(replica_dir), bias=bias, seed=pdb_seed)
            
            # Find trajectory file - prefer COLVAR for CV-based evaluation
            trajectory_file = replica_dir / "COLVAR"
            if not trajectory_file.exists():
                trajectory_file = replica_dir / "output.pdb"
            
            if not trajectory_file.exists():
                raise FileNotFoundError(f"No trajectory found in {replica_dir}")
            
            # Evaluate this replica
            replica_score = evaluator.evaluate_from_file(str(trajectory_file))
            scores.append(replica_score)
        
        # Average over replicas
        score = np.mean(scores)
        score_std = np.std(scores) if n_replicas > 1 else 0.0
        
        elapsed = time.time() - start_time
        
        if n_replicas > 1:
            print(f"  [{gen:03d}.{ind:03d}] Score: {score:.4f} ± {score_std:.4f} (n={n_replicas}, {elapsed:.1f}s)")
        else:
            print(f"  [{gen:03d}.{ind:03d}] Score: {score:.4f} ({elapsed:.1f}s)")
        
        return (ind, normalized_params, score, elapsed)
        
    except Exception as e:
        print(f"  [{gen:03d}.{ind:03d}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return (ind, normalized_params, 1e6, time.time() - start_time)


class CMAESWorkflow:
    """CMA-ES workflow that orchestrates bias parameter optimization.
    
    Manages the complete optimization loop including:
    - CMA-ES ask/tell interface
    - Parallel simulation execution
    - Evaluation and scoring
    - Checkpointing and resuming
    
    Connects: bias ← workflow → sampler → evaluator
    """
    
    def __init__(
        self,
        bias,
        sampler,
        evaluator,
        initial_mean=None,
        sigma=0.3,
        population_size=None,
        bounds=None,
        max_generations=20,
        n_workers=1,  # Number of parallel workers
        n_replicas=1,  # Number of replicas per evaluation
        early_stop_patience=50,  # Early stopping patience
        early_stop_threshold=0.01  # Early stopping threshold
    ):
        """
        Args:
            bias: Bias object with parameter interface
            sampler: Sampler with .run(output_dir, bias) method
            evaluator: Evaluator with .evaluate_from_file(trajectory_file) method
            initial_mean: Starting point (if None, use bias defaults)
            sigma: Initial step size
            population_size: CMA-ES population (if None, auto-set)
            bounds: Parameter bounds (if None, use bias bounds)
            max_generations: Maximum iterations
            n_workers: Number of parallel workers (1 = serial)
            n_replicas: Number of replicas per evaluation to average (default 1)
            early_stop_patience: Stop if no improvement for N generations (0 = disabled)
            early_stop_threshold: Relative improvement threshold for early stopping
        """
        self.bias = bias
        self.sampler = sampler
        self.evaluator = evaluator
        self.max_generations = max_generations
        self.n_workers = n_workers
        self.n_replicas = n_replicas
        self.early_stop_patience = early_stop_patience
        self.early_stop_threshold = early_stop_threshold
        
        # Store class and construction info for multiprocessing
        self._bias_info = self._get_construction_info(bias) if bias is not None else None
        self._sampler_info = self._get_construction_info(sampler) if sampler is not None else None
        self._evaluator_info = self._get_construction_info(evaluator)
        
        # Get parameter space from bias
        n_params = bias.get_parameter_space_size()
        
        # Use bias defaults if not provided
        if bounds is None:
            bounds = bias.get_parameter_bounds()
        
        # Ensure bounds are (n_params, 2)
        if bounds.shape != (n_params, 2):
            raise ValueError(
                f"Bounds must have shape ({n_params}, 2), got {bounds.shape}"
            )
        
        # Store for denormalization
        self.bounds = bounds
        
        # Get initial parameters
        if initial_mean is None:
            # Normalize bias defaults to [0, 1]
            defaults = bias.get_default_parameters()
            
            # Normalize to [0, 1]
            range_size = bounds[:, 1] - bounds[:, 0]
            initial_mean = (defaults - bounds[:, 0]) / range_size
        
        # Validate initial_mean
        initial_mean = np.clip(initial_mean, 0.0, 1.0)
        
        if population_size is None:
            population_size = 4 + int(3 * np.log(n_params))  # CMA-ES default
        
        print(f"\nCreating CMA-ES optimizer:")
        print(f"  Parameters: {n_params}")
        print(f"  Population: {population_size}")
        print(f"  Sigma: {sigma}")
        print(f"  Workers: {self.n_workers}")
        if self.n_replicas > 1:
            print(f"  Replicas/eval: {self.n_replicas} (reduces stochastic noise)")
            total_sims = max_generations * population_size * self.n_replicas
            print(f"  Total simulations: ~{total_sims}")
        
        # Use Separable CMA-ES (diagonal covariance only) - matches Alberto's implementation
        # This is better for problems where parameters are mostly independent
        self.cma = SepCMA(
            mean=initial_mean,
            sigma=sigma,
            population_size=population_size,
            bounds=np.array([[0.0, 1.0]] * n_params)        
            )
        print("✅ SepCMA-ES created successfully (diagonal covariance)")
    
    def _get_construction_info(self, obj):
        """Extract class and constructor kwargs from object.
        
        For objects to be reconstructed in worker processes.
        
        Returns:
            (class, kwargs_dict)
        """
        obj_class = obj.__class__
        
        # Try to get constructor signature to know what arguments are valid
        import inspect
        try:
            sig = inspect.signature(obj_class.__init__)
            valid_params = set(sig.parameters.keys()) - {'self'}
        except:
            valid_params = None  # Fallback: include all attributes
        
        kwargs = {}
        
        # Get all attributes
        for key, value in obj.__dict__.items():
            # Skip private attributes
            if key.startswith('_'):
                continue
            
            # Only include if it's a constructor parameter
            if valid_params is not None and key not in valid_params:
                continue
            
            # Only include simple types that can be pickled
            if isinstance(value, (int, float, str, bool, tuple, type(None))):
                kwargs[key] = value
            elif isinstance(value, np.ndarray):
                kwargs[key] = value
            elif isinstance(value, list):
                # Check if list contains only simple types
                if all(isinstance(item, (int, float, str, bool)) for item in value):
                    kwargs[key] = value
        
        return (obj_class, kwargs)
    
    def _denormalize(self, normalized: np.ndarray) -> np.ndarray:
        """Convert [0,1] to actual parameter values."""
        normalized = np.clip(normalized, 0.0, 1.0)
        return self.bounds[:, 0] + normalized * (self.bounds[:, 1] - self.bounds[:, 0])
    
    def _save_checkpoint(self, history: list, output_dir: str):
        """Save current optimization state to a checkpoint file."""
        import pickle
        
        # Get best solution from history so far
        all_best_scores = [h['best_score'] for h in history]
        best_gen_idx = int(np.argmin(all_best_scores))
        best_score = all_best_scores[best_gen_idx]
        best_normalized = history[best_gen_idx]['best_solution']
        best_params = self._denormalize(best_normalized)
        
        # Save complete CMA-ES state for resuming (use built-in pickle support)
        cma_state = self.cma.__getstate__()
        
        checkpoint = {
            'best_parameters': best_params,
            'best_score': best_score,
            'best_generation': best_gen_idx,
            'history': history,
            'cma_state': cma_state,
            'cma_mean': self.cma.mean,  # Keep for backwards compatibility
            'latest_generation': history[-1]['generation'],
            # Save optimizer settings for proper resume
            'population_size': self.cma.population_size,
            'n_workers': self.n_workers,
            'n_replicas': self.n_replicas,
            'sigma': self.cma._sigma,
            'max_generations': self.max_generations
        }
        
        checkpoint_file = Path(output_dir) / "optimization_checkpoint.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        return checkpoint_file
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and restore optimization state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Tuple of (history, generation, cma_state)
        """
        import pickle
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        print(f"\n📂 Loading checkpoint from: {checkpoint_path}")
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        history = checkpoint['history']
        latest_gen = checkpoint['latest_generation']
        cma_state = checkpoint.get('cma_state', None)
        
        print(f"✅ Loaded {len(history)} generations (last: {latest_gen})")
        print(f"   Best score so far: {checkpoint['best_score']:.4f} (gen {checkpoint['best_generation']})")
        
        return history, latest_gen, cma_state
    
    def _restore_cma_state(self, cma_state: dict):
        """Restore CMA-ES optimizer state from checkpoint.
        
        Args:
            cma_state: Dictionary containing CMA-ES state
        """
        if cma_state is None:
            print("⚠️  Warning: No CMA state in checkpoint, starting fresh CMA-ES")
            return
        
        print("🔄 Restoring CMA-ES state...")
        
        # Use built-in __setstate__ to restore complete internal state
        self.cma.__setstate__(cma_state)
        
        print(f"   Mean: {self.cma.mean[:3]}...")
        print(f"   Sigma: {self.cma._sigma:.4f}")
        print(f"   Generation: {self.cma.generation}")
    
    def optimize(self, output_dir: str = "output/cmaes", resume_from: str = None) -> dict:
        """Run CMA-ES optimization loop with parallel evaluations.
        
        Automatically detects if evaluator requires simulation or can work analytically.
        
        Args:
            output_dir: Directory to save results
            resume_from: Path to checkpoint file to resume from (optional)
        
        Returns:
            Dict with best_parameters, best_score, history
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Check if evaluator needs simulation
        analytical_mode = not self.evaluator.requires_simulation
        
        # Handle checkpoint resuming
        if resume_from is not None:
            history, last_gen, cma_state = self._load_checkpoint(resume_from)
            generation = last_gen + 1
            self._restore_cma_state(cma_state)
            
            # Initialize early stopping from history
            all_scores = [h['best_score'] for h in history]
            best_ever_score = min(all_scores)
            
            # Count generations without improvement from end of history
            generations_without_improvement = 0
            for i in range(len(history) - 1, -1, -1):
                if history[i]['best_score'] <= best_ever_score * (1 + self.early_stop_threshold):
                    break
                generations_without_improvement += 1
        else:
            history = []
            generation = 0
            best_ever_score = float('inf')
            generations_without_improvement = 0
        
        print("\n" + "="*60)
        mode = 'Serial' if self.n_workers == 1 else f'Parallel ({self.n_workers} workers)'
        eval_mode = 'Analytical' if analytical_mode else 'MD Simulation'
        resume_msg = f" (Resuming from gen {generation})" if resume_from else ""
        print(f"CMA-ES Optimization ({mode}, {eval_mode}){resume_msg}")
        print("="*60)
        
        if analytical_mode:
            print("  ⚡ Analytical evaluation - no MD simulation needed!")
        
        if resume_from:
            print(f"  🔄 Resumed from generation {generation - 1}")
            print(f"  📊 Starting with {len(history)} completed generations")
        
        if self.early_stop_patience > 0:
            print(f"  Early stopping: enabled (patience={self.early_stop_patience}, threshold={self.early_stop_threshold:.1%})")
        
        # Create process pool if using multiprocessing
        if self.n_workers > 1:
            executor = ProcessPoolExecutor(max_workers=self.n_workers)
        else:
            executor = None
        
        # Track if we need to return early
        interrupted = False
        
        try:
            while generation < self.max_generations and not self.cma.should_stop():
                gen_start = time.time()
                print(f"\n--- Generation {generation} ---")
                
                # Clear generation directory if it exists (from incomplete previous run)
                # This ensures we don't append to partial data when resuming
                if not analytical_mode:
                    gen_dir = Path(output_dir) / f"gen{generation:03d}"
                    if gen_dir.exists():
                        import shutil
                        print(f"  🧹 Cleaning existing generation directory: {gen_dir.name}")
                        shutil.rmtree(gen_dir)
                
                # Get population from CMA-ES
                solutions = []
                for _ in range(self.cma.population_size):
                    x = self.cma.ask()
                    # Verify whether in 0-1 bounds
                    if np.any(x < 0.0) or np.any(x > 1.0):
                        print(f"  ⚠️  Warning: Sampled parameter out of bounds: {x}")
                        raise ValueError("CMA-ES sampled parameter out of [0, 1] bounds")
                    solutions.append(x)
                
                # Prepare arguments for workers
                worker_args = []
                for i, x in enumerate(solutions):
                    params = self._denormalize(x)
                    
                    if analytical_mode:
                        # Analytical evaluation - pass evaluator directly (can't pickle functions)
                        args = (
                            self.evaluator,  # Pass object directly
                            params,
                            x,
                            generation,
                            i
                        )
                    else:
                        # MD simulation evaluation - pack all construction info
                        args = (
                            self._bias_info[0], self._bias_info[1],
                            self._sampler_info[0], self._sampler_info[1],
                            self._evaluator_info[0], self._evaluator_info[1],
                            params,
                            x,
                            generation,
                            i,
                            output_dir,
                            self.n_replicas
                        )
                    worker_args.append(args)
                
                # Evaluate population - use appropriate worker function
                worker_func = _evaluate_analytical_worker if analytical_mode else _evaluate_worker
                
                if self.n_workers == 1:
                    # Serial execution
                    results = [worker_func(args) for args in worker_args]
                else:
                    # Parallel execution
                    print(f"  Submitting {len(worker_args)} jobs to {self.n_workers} workers...")
                    futures = [executor.submit(worker_func, args) for args in worker_args]
                    results = [future.result() for future in futures]
                
                # Sort results by individual index to maintain order
                results = sorted(results, key=lambda r: r[0])
                
                # Extract results
                evaluated = [(x, score) for _, x, score, _ in results]
                times = [t for _, _, _, t in results]
                
                # Update CMA-ES
                self.cma.tell(evaluated)
                
                # Track history
                scores = [score for _, score in evaluated]
                best_score = min(scores)
                mean_score = np.mean(scores)
                best_idx = np.argmin(scores)
                best_solution = solutions[best_idx]
                
                gen_elapsed = time.time() - gen_start
                
                # Construct generation output directory
                if not analytical_mode:
                    gen_output_dir = str(Path(output_dir) / f"gen{generation:03d}")
                else:
                    gen_output_dir = None
                history_entry = {
                    'generation': generation,
                    'best_score': best_score,
                    'mean_score': mean_score,
                    'std_score': np.std(scores),
                    'all_scores': scores,
                    'best_solution': best_solution,
                    'eval_times': times,
                    'generation_time': gen_elapsed,
                    'population': solutions,  # All normalized solutions
                    'cma_mean': self.cma.mean.copy(),  # CMA-ES mean
                    'cma_sigma': self.cma._sigma,  # CMA-ES sigma (step size)
                    'cma_cov': self.cma._C.copy() if hasattr(self.cma, '_C') else None,  # Covariance
                    'output_dir': gen_output_dir  # Path to generation output
                }
                history.append(history_entry)
                
                print(f"\nGeneration {generation} complete ({gen_elapsed:.1f}s):")
                print(f"  Best:  {best_score:.4f}")
                print(f"  Mean:  {mean_score:.4f}")
                print(f"  Std:   {np.std(scores):.4f}")
                print(f"  Avg eval time: {np.mean(times):.1f}s")
                if self.n_workers > 1:
                    speedup = sum(times) / gen_elapsed
                    efficiency = speedup / self.n_workers * 100
                    print(f"  Speedup: {speedup:.2f}x ({efficiency:.1f}% efficiency)")
                
                # Check for early stopping
                if self.early_stop_patience > 0:
                    # Check if current best is better than all-time best
                    if best_score < best_ever_score:
                        # Calculate relative improvement
                        relative_improvement = (best_ever_score - best_score) / (abs(best_ever_score) + 1e-10)
                        
                        # Only reset counter if improvement is significant (above threshold)
                        if relative_improvement > self.early_stop_threshold:
                            best_ever_score = best_score
                            generations_without_improvement = 0
                            print(f"  ✓ Improvement: {relative_improvement:.2%} (new best: {best_score:.2f})")
                        else:
                            # Small improvement, don't reset counter but update best
                            best_ever_score = best_score
                            generations_without_improvement += 1
                            print(f"  → Tiny improvement: {relative_improvement:.3%} ({generations_without_improvement}/{self.early_stop_patience})")
                    else:
                        # No improvement this generation
                        generations_without_improvement += 1
                        print(f"  ⏸  No improvement ({generations_without_improvement}/{self.early_stop_patience})")
                        
                        if generations_without_improvement >= self.early_stop_patience:
                            print(f"\n⚠️  Early stopping: No significant improvement for {self.early_stop_patience} generations")
                            print(f"     Best score achieved: {best_ever_score:.4f}")
                            break
                
                generation += 1
                
                # Save checkpoint after each generation
                checkpoint_file = self._save_checkpoint(history, output_dir)
                print(f"  💾 Checkpoint saved: {checkpoint_file.name}")
                
        except KeyboardInterrupt:
            print("\n\n" + "="*60)
            print("⚠️  KEYBOARD INTERRUPT - Saving progress...")
            print("="*60)
            interrupted = True
            if len(history) > 0:
                checkpoint_file = self._save_checkpoint(history, output_dir)
                print(f"✅ Progress saved to: {checkpoint_file}")
            else:
                print("⚠️  No generations completed - nothing to save")
        finally:
            # Clean up process pool
            if executor is not None:
                executor.shutdown(wait=True)
        
        # Get final best solution from entire history
        if len(history) == 0:
            print("\n⚠️  No generations completed - cannot return results")
            return None
            
        all_best_scores = [h['best_score'] for h in history]
        best_gen_idx = int(np.argmin(all_best_scores))
        best_score = all_best_scores[best_gen_idx]
        best_normalized = history[best_gen_idx]['best_solution']
        best_params = self._denormalize(best_normalized)
        
        print("\n" + "="*60)
        if interrupted:
            print(f"Optimization interrupted after {len(history)} generations")
        elif self.cma.should_stop():
            print(f"CMA-ES convergence detected after {len(history)} generations")
            print(f"  Reason: CMA-ES internal stopping criterion met")
            print(f"  Current sigma: {self.cma._sigma:.6f}")
            print(f"  (Sigma < tolerance or search space collapsed)")
        else:
            print(f"Optimization complete!")
        print(f"  Best score: {best_score:.4f} (generation {best_gen_idx})")
        print(f"  Best params: {best_params}")
        print("="*60)
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'best_generation': best_gen_idx,
            'history': history,
            'cma_mean': self.cma.mean,
            'interrupted': interrupted
        }