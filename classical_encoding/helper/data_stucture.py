# for same issue as https://github.com/tiangolo/typer/issues/348,
# still need to use Optional["BinaryTree"]
from typing import Optional


class BinaryTreeNode:
    # binary tree node, known as MetaSymbol in Huffman coding
    value: Optional[int]
    left: Optional["BinaryTreeNode"]
    right: Optional["BinaryTreeNode"]

    def __init__(
        self,
        value: Optional[int] = None,
        left: Optional["BinaryTreeNode"] = None,
        right: Optional["BinaryTreeNode"] = None,
    ):
        self.left = left
        self.right = right
        self.value = value

    # #TODO: serialize tree


class RestrictedFastOrderedList:
    """
    Manage an ordered list of integers with a dictionary mapping integers to
    their index ranges in the list. Only supports incrementing elements by one
    and adding new elements with value 0.

    Used for FGK algorithm in adaptive Huffman coding.
    """

    __l: list[int]  # The ordered list of integers, from greatest to least
    __d: dict[int, tuple[int, int]]  #  map integers to the start and end indices

    @property
    def dict(self) -> dict:
        """
        Return a copy of the current state of the dictionary.

        Returns:
            dict: The current dictionary mapping integers to their index ranges.
        """
        return dict(self.__d)

    @property
    def list(self) -> list:
        """
        Return a copy of the current state of the list.

        Returns:
            list: The current ordered list of integers.
        """
        return list(self.__l)

    def __init__(self):
        """Initialize an empty list and a dictionary for tracking indices."""
        self.__l = []
        self.__d = {}

    def add_one(self, index: int) -> tuple[int, int]:
        """Increment the element at the given index by one and update the list
        and dictionary.

        Args:
            index (int): The index of the element to be incremented.

        Returns:
            tuple[int, int]: The indices of the swapped pair (old index, new index)
        """
        if index < 0 or index >= len(self.__l):
            raise IndexError("Index out of range")

        old_value = self.__l[index]
        new_value = old_value + 1

        # Update the list
        start, end = self.__d[old_value]  # start will be the returned new index
        self.__l[index] = new_value
        self.__l[index], self.__l[start] = self.__l[start], self.__l[index]

        # Update the dictionary for the old value
        if end - start == 1:
            del self.__d[old_value]
        else:
            self.__d[old_value] = (start + 1, end)

        # Update the dictionary for the new value
        if new_value in self.__d:
            new_start, new_end = self.__d[new_value]
            assert new_end == start, f"new_end: {new_end}, start: {start}"
            self.__d[new_value] = (new_start, new_end + 1)
        else:
            self.__d[new_value] = (start, start + 1)

        return (index, start)

    def new_item(self) -> int:
        """Add a new element with value 0 to the list and return its index.

        Returns:
            int: The index of the new element.
        """
        value = 0
        if value in self.__d:
            start, end = self.__d[value]
            self.__l.append(value)  # Insert at the end of the group
            self.__d[value] = (start, end + 1)
            return end
        else:
            self.__l.append(value)
            self.__d[value] = (len(self.__l) - 1, len(self.__l))
            return len(self.__l) - 1

    def _check(self):
        """Check if the list and dictionary are consistent."""
        last_end = 0  # the first start should be 0
        for k, (start, end) in sorted(
            self.__d.items(), key=lambda x: x[0], reverse=True
        ):
            assert start < end
            assert start == last_end
            assert self.__l[start:end] == [k] * (end - start)
            last_end = end

    def __len__(self) -> int:
        return len(self.__l)

    def __iter__(self):
        return iter(self.__l)

    def __getitem__(self, index: int) -> int:
        return self.__l[index]

    def __str__(self) -> str:
        self._check()
        return f"OrderedList({self.__l})"

    def __repr__(self):
        return f"RestrictedFastOrderedList({self.__l}, {self.__d})"


class ExtendedRestrictedFastOrderedList[T]:
    """
    Extended version of RestrictedFastOrderedList that supports dependency injection
    of items of type T. Manages an ordered list of integers paired with instances of type T.
    """

    __instance_index: dict[T, int]  #  {T: index}
    __ordered_instances: list[T]  # The ordered list of instances of T
    __fast_ordered_list: RestrictedFastOrderedList

    @property
    def instance_index(self) -> dict[T, int]:
        """
        Return a copy of the current state of the dictionary.
        """
        return dict(self.__instance_index)

    @property
    def ordered_instances(self) -> list[T]:
        """
        Return a copy of the current state of the list.
        """
        return list(self.__ordered_instances)

    @property
    def instance_weight(self) -> dict[T, int]:
        """
        Return the weight of each instance.
        """
        return {
            instance: self.__fast_ordered_list[index]
            for instance, index in self.__instance_index.items()
        }

    def __init__(self):
        self.__instance_index = {}
        self.__ordered_instances = []
        self.__fast_ordered_list = RestrictedFastOrderedList()

    def add_one(self, instance: T) -> tuple[T, T]:
        """
        Increment the weight paired with the given instance by one and update
        the list and dictionary.

        Args:
            instance (T): The instance whose paired weight is to be incremented.

        Returns:
            tuple[T, T]: The instances corresponding to the swapped pair.
        """
        if instance not in self.__instance_index:
            raise ValueError("Instance not found in the list.")

        index = self.__instance_index[instance]
        old_index, new_index = self.__fast_ordered_list.add_one(index)
        swapped_instance = self.__ordered_instances[new_index]

        # update the instance order
        (
            self.__ordered_instances[new_index],
            self.__ordered_instances[old_index],
        ) = (
            self.__ordered_instances[old_index],
            self.__ordered_instances[new_index],
        )

        # update the instance index
        (
            self.__instance_index[self.__ordered_instances[new_index]],
            self.__instance_index[self.__ordered_instances[old_index]],
        ) = (
            self.__instance_index[self.__ordered_instances[old_index]],
            self.__instance_index[self.__ordered_instances[new_index]],
        )

        # return the swapped instances
        return (instance, swapped_instance)

    def new_item(self, instance: T):
        """
        Add a new element with weight 0 paired with the given instance to the list.

        Args:
            instance (T): The instance to be added.
        """
        if instance in self.__instance_index:
            raise ValueError("Instance already exists in the list")

        index = self.__fast_ordered_list.new_item()
        print(self.__ordered_instances)
        print(len(self.__ordered_instances))
        assert index == len(
            self.__ordered_instances
        ), f"{index=} != {len(self.__ordered_instances)=}, {self.__ordered_instances=}"
        self.__instance_index[instance] = index
        self.__ordered_instances.append(instance)

    def _check(self):
        """Check if the list and dictionary are consistent."""
        self.__fast_ordered_list._check()  # check the fast ordered list
        # check the instance index and the ordered instances
        assert len(self.__ordered_instances) == len(self.__instance_index)
        for i, instance in enumerate(self.__ordered_instances):
            assert self.__instance_index[instance] == i

    def __len__(self) -> int:
        return len(self.__ordered_instances)

    def __iter__(self):
        return iter(self.__ordered_instances)

    def __getitem__(self, index: int) -> T:
        return self.__ordered_instances[index]

    def __str__(self) -> str:
        self._check()
        return f"ExtendedRestrictedFastOrderedList({self.__ordered_instances})"

    def __repr__(self):
        return "ExtendedRestrictedFastOrderedList(NOT_IMPLEMENTED)"


from typing import Optional, Literal

# easier Enum
BirthOrder = Literal[1, 0]
BIRTH_ORDER_LEFT = 1
BIRTH_ORDER_RIGHT = 0


class SwappableNode[T]:
    parent: "SwappableNode"
    birth_order: BirthOrder
    left: Optional["SwappableNode"]
    right: Optional["SwappableNode"]
    value: T

    def __init__(
        self,
        value: T,
        parent: Optional["SwappableNode"] | None = None,
        birth_order: BirthOrder | None = None,
        left: Optional["SwappableNode"] = None,
        right: Optional["SwappableNode"] = None,
    ):
        self.value = value
        self.parent = parent if parent is not None else self
        self.birth_order = birth_order if birth_order is not None else BIRTH_ORDER_LEFT
        self.left = left
        self.right = right

        # Update parent's child references
        if self.birth_order == BIRTH_ORDER_LEFT:
            if self.parent.left is not None:
                raise ValueError("Parent already has a left child")
            self.parent.left = self
        else:
            if self.parent.right is not None:
                raise ValueError("Parent already has a right child")
            self.parent.right = self

    def swap(self, other: "SwappableNode[T]"):
        # Check if either node is the parent of the other
        if self.parent == other or other.parent == self:
            raise ValueError("Cannot swap a node with its direct parent")

        # Swap birth orders and parents
        self.birth_order, other.birth_order = other.birth_order, self.birth_order
        self.parent, other.parent = other.parent, self.parent

        # Update parent's child references
        if self.birth_order == BIRTH_ORDER_LEFT:
            self.parent.left = self
        else:
            self.parent.right = self
        if other.birth_order == BIRTH_ORDER_LEFT:
            other.parent.left = other
        else:
            other.parent.right = other


def test_restricted_fast_ordered_list():
    manager = RestrictedFastOrderedList()

    # Adding new items and incrementing values
    index1 = manager.new_item()
    _index2 = manager.add_one(index1)
    index3 = manager.new_item()
    _, index4 = manager.add_one(index3)
    _, index5 = manager.add_one(index4)
    manager.add_one(index5)
    index3 = manager.new_item()
    manager.add_one(2)
    manager.add_one(2)
    manager.add_one(2)
    manager.add_one(2)
    manager.add_one(2)
    manager.add_one(2)
    # #TODO: add more test cases

    # Example showing error handling
    try:
        manager.add_one(10)  # Assuming 10 is an out-of-range index
    except IndexError:
        pass  # print("\nExpected Error:", e)
    else:
        raise AssertionError("Expected IndexError")


def test_extended_restricted_fast_ordered_list():
    class Example:
        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return f"Example({self.name})"

    extended_manager = ExtendedRestrictedFastOrderedList[Example]()

    # Demonstration of adding new items and incrementing values
    instance_a = Example("A")
    instance_b = Example("B")
    instance_c = Example("C")

    extended_manager.new_item(instance_a)
    extended_manager.new_item(instance_b)
    extended_manager.add_one(instance_a)
    for _ in range(3):
        extended_manager.add_one(instance_b)
    extended_manager.new_item(instance_c)
    for _ in range(5):
        extended_manager.add_one(instance_a)
    for _ in range(2):
        extended_manager.add_one(instance_c)
    for _ in range(8):
        extended_manager.add_one(instance_c)


def test_swappable_node():
    dummy_root = SwappableNode(None)  # dummy root is its own left child
    root = SwappableNode(0, dummy_root, BIRTH_ORDER_RIGHT)
    node1 = SwappableNode(1, root, BIRTH_ORDER_LEFT)
    node2 = SwappableNode(2, root, BIRTH_ORDER_RIGHT)
    node3 = SwappableNode(3, node1, BIRTH_ORDER_LEFT)
    node4 = SwappableNode(4, node1, BIRTH_ORDER_RIGHT)
    node5 = SwappableNode(5, node2, BIRTH_ORDER_LEFT)
    node6 = SwappableNode(6, node2, BIRTH_ORDER_RIGHT)
    node7 = SwappableNode(7, node3, BIRTH_ORDER_LEFT)
    node8 = SwappableNode(8, node3, BIRTH_ORDER_RIGHT)
    _node9 = SwappableNode(9, node4, BIRTH_ORDER_LEFT)
    _node10 = SwappableNode(10, node4, BIRTH_ORDER_RIGHT)
    _node11 = SwappableNode(11, node5, BIRTH_ORDER_LEFT)
    _node12 = SwappableNode(12, node5, BIRTH_ORDER_RIGHT)
    _node13 = SwappableNode(13, node6, BIRTH_ORDER_LEFT)
    _node14 = SwappableNode(14, node6, BIRTH_ORDER_RIGHT)
    node15 = SwappableNode(15, node7, BIRTH_ORDER_LEFT)
    node16 = SwappableNode(16, node7, BIRTH_ORDER_RIGHT)
    _node17 = SwappableNode(17, node8, BIRTH_ORDER_LEFT)
    _node18 = SwappableNode(18, node8, BIRTH_ORDER_RIGHT)
    try:
        node2.swap(node6)
    except ValueError:
        pass  # Expected "Cannot swap a node with its direct parent"
    else:
        raise AssertionError("Expected ValueError")
    node2.swap(node7)
    assert node7.parent == root
    assert node7.birth_order == BIRTH_ORDER_RIGHT
    assert node2.parent == node3
    assert node2.birth_order == BIRTH_ORDER_LEFT
    assert root.left == node1
    assert root.right == node7
    assert node2.left == node5
    assert node2.right == node6
    assert node7.left == node15
    assert node7.right == node16

    print("swap passed")


if __name__ == "__main__":
    test_restricted_fast_ordered_list()
    test_extended_restricted_fast_ordered_list()
    test_swappable_node()
