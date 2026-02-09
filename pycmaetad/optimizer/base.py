from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def ask(self):
        pass
    @abstractmethod
    def tell(self, solutions):
        pass