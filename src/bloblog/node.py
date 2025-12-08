from collections.abc import Buffer
from abc import abstractmethod, ABC
from .pubsub import Pub, Sub
from dataclasses import dataclass

class Codec[T](ABC):
    @abstractmethod
    def encode(self, item: T) -> Buffer:
        ...
    
    @abstractmethod
    def decode(self, data: Buffer) -> T:
        ...

@dataclass
class NodeSpec:
    node: "Node"
    inputs: list["ChannelSpec"]
    outputs: list["ChannelSpec"]

@dataclass
class ChannelSpec[T]:
    name: str
    codec: Codec[T]

class Node(ABC):
    @abstractmethod
    def get_spec(self) -> "NodeSpec":
        ...

    @abstractmethod
    async def process(self, inputs: list[Sub], outputs: list[Pub]) -> None:
        ...


