## Dev

```bash
poetry install
poetry shell
```

## Note

1. Big Endian across the whole repo.

## #TODO

- [ ] Introduction to every algorithm
- [ ] Flexible symbol size like 1 bit or 2 bits, now it's fixed to 1 byte
- [ ] Show the compression ratio
- [ ] N-ary Huffman coding
- [ ] Add a transmitter to the pipeline and implement error correction
- [ ] UML graph of algorithms

### Small

- [ ] Use constant LEFT, RIGHT instead of 0, 1 or 0, 1

## Note for FGK algorithm

1. ending symbol like EOT/EOF
2. A way to represent bytes in bits
3. On updating the tree for unseen symbol, one should not swap the current node with it's parent, although it's parent is the "first with same weight" node. Instead, do it with methods like `new_node()` and `add_one()` from `RestrictedFastOrderedList` directly.
4. On updating the tree for seen symbol, we should not use the static parent.
5. When there are only three nodes with weight `<=W` (with smallest `W`, the third last element in the weight list), they should be

   1. NYT node with weight 0;
   2. Symbol node with weight `W`;
   3. The common direct parent of NYT and the symbol node.

   In this case, when the algorithm try to swap the symbol node with "first with same weight" node, because swapping with parent node is not allowed, the algorithm would stopped with an error.
