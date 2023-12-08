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
- [ ] Test with more probability distributions of symbols for to find the relation with entropy

### Small

- [ ] Use constant LEFT, RIGHT instead of 0, 1 or 0, 1

## Note for FGK algorithm

1. ending symbol like EOT/EOF
2. A way to represent bytes in bits
3. On updating the tree for unseen symbol, one should not swap the current node with it's parent, although it's parent is the "first with same weight" node. Instead, do it with methods like `new_node()` and `add_one()` from `RestrictedFastOrderedList` directly.
4. On updating the tree for seen symbol, we should not use the static parent.
5. (swap NYT parent) When there are only three nodes with weight `<=W` (with smallest `W`, the third last element in the weight list), they should be

   1. NYT node with weight 0;
   2. Symbol node with weight `W`;
   3. The common direct parent of NYT and the symbol node.

   In this case, when the algorithm try to swap the symbol node with "first with same weight" node, because swapping with parent node is not allowed, the algorithm would stopped with an error.
   I found a way to avoid this: Just add_one to the both element with weight `W` without swapping them, and continue the update algorithm with the grandparent of the NYT node.

6. (wrong add_one) Another edge case with `./edge-case/wrong add_one.binary`, caused by the fix of the previous edge case "swap NYT parent". The tree where the problem happened is like this

   ```text
   M(2238)[M(906)[M(410)[M(193)[M(96)[M(48)[M(24)[M(12)[214(6),80(6)],M(12)[81(6),231(6)]],M(24)[27(12),M(12)[50(6),66(6)]]],M(48)[M(24)[M(12)[48(6),69(6)],241(12)],M(24)[203(12),M(12)[148(6),160(6)]]]],M(97)[M(48)[M(24)[114(12),243(12)],M(24)[M(12)[229(6),255(6)],109(12)]],M(49)[M(24)[136(12),71(12)],M(25)[18(12),25(13)]]]],M(217)[M(105)[M(52)[M(26)[196(13),122(13)],M(26)[106(13),32(13)]],M(53)[M(26)[52(13),218(13)],M(27)[164(13),M(14)[21(7),226(7)]]]],M(112)[M(56)[M(28)[212(14),253(14)],M(28)[M(14)[87(7),222(7)],89(14)]],M(56)[M(28)[230(14),M(14)[168(7),M(7)[73(3),146(4)]]],M(28)[126(14),128(14)]]]]],M(496)[M(235)[M(112)[M(56)[M(28)[M(14)[6(7),34(7)],165(14)],M(28)[M(14)[219(7),117(7)],M(14)[10(7),213(7)]]],M(56)[M(28)[M(14)[M(7),133(7)],M(14)[197(7),202(7)]],M(28)[M(14)[236(7),110(7)],254(14)]]],M(123)[M(59)[M(28)[M(14)[154(7),233(7)],M(14)[56(7),115(7)]],M(31)[M(15)[139(7),124(8)],M(16)[130(8),161(8)]]],M(64)[M(32)[M(16)[91(8),16(8)],M(16)[104(8),248(8)]],M(32)[M(16)[250(8),5(8)],M(16)[249(8),125(8)]]]]],M(261)[M(128)[M(64)[M(32)[M(16)[174(8),228(8)],M(16)[107(8),76(8)]],M(32)[M(16)[M(8)[7(4),93(4)],162(8)],M(16)[M(8)[184(4),185(4)],77(8)]]],M(64)[M(32)[M(16)[156(8),140(8)],M(16)[M(8)[179(4),101(4)],199(8)]],M(32)[M(16)[36(8),186(8)],M(16)[191(8),M(8)[60(4),86(4)]]]]],M(133)[M(64)[M(32)[M(16)[181(8),119(8)],M(16)[65(8),152(8)]],M(32)[M(16)[166(8),123(8)],M(16)[79(8),201(8)]]],M(69)[M(33)[M(16)[54(8),99(8)],14(17)],M(36)[M(18)[23(9),198(9)],M(18)[63(9),13(9)]]]]]]],M(1332)[M(607)[M(288)[M(144)[M(72)[M(36)[M(18)[20(9),157(9)],M(18)[92(9),227(9)]],M(36)[M(18)[220(9),215(9)],M(18)[187(9),43(9)]]],M(72)[M(36)[M(18)[90(9),83(9)],M(18)[221(9),134(9)]],M(36)[M(18)[38(9),49(9)],M(18)[17(9),97(9)]]]],M(144)[M(72)[M(36)[1(18),M(18)[51(9),113(9)]],M(36)[M(18)[145(9),216(9)],M(18)[239(9),68(9)]]],M(72)[M(36)[M(18)[163(9),58(9)],M(18)[142(9),240(9)]],M(36)[M(18)[53(9),200(9)],M(18)[173(9),238(9)]]]]],M(319)[M(159)[M(79)[M(39)[151(19),M(20)[131(10),M(10)[45(5),70(5)]]],M(40)[M(20)[88(10),82(10)],M(20)[204(10),235(10)]]],M(80)[M(40)[M(20)[59(10),247(10)],M(20)[64(10),62(10)]],M(40)[M(20)[61(10),M(10)[189(5),150(5)]],M(20)[94(10),M(10)[149(5),24(5)]]]]],M(160)[M(80)[M(40)[M(20)[M(10)[171(5),223(5)],2(10)],M(20)[245(10),M(10)[144(5),55(5)]]],M(40)[M(20)[M(10)[135(5),205(5)],147(10)],M(20)[M(10)[67(5),M(5)[M(2)[M(1)[256(0),210(1)],224(2)],39(3)]],M(10)[33(5),35(5)]]]],M(80)[M(40)[M(20)[29(10),22(10)],M(20)[M(10)[167(5),232(5)],180(10)]],M(40)[M(20)[190(10),M(10)[129(5),170(5)]],M(20)[95(10),244(10)]]]]]],M(725)[M(347)[M(171)[M(83)[M(40)[M(20)[M(10)[15(5),26(5)],44(10)],M(20)[153(10),74(10)]],M(43)[M(21)[137(10),102(11)],M(22)[28(11),121(11)]]],M(88)[M(44)[M(22)[78(11),72(11)],M(22)[251(11),127(11)]],M(44)[M(22)[41(11),116(11)],M(22)[37(11),111(11)]]]],M(176)[M(88)[M(44)[M(22)[155(11),3(11)],M(22)[188(11),105(11)]],M(44)[M(22)[57(11),234(11)],M(22)[84(11),100(11)]]],M(88)[M(44)[M(22)[169(11),192(11)],M(22)[31(11),75(11)]],M(44)[M(22)[183(11),194(11)],M(22)[9(11),42(11)]]]]],M(378)[M(186)[M(90)[M(44)[M(22)[237(11),172(11)],M(22)[207(11),217(11)]],M(46)[M(22)[211(11),176(11)],M(24)[206(12),M(12)[177(6),19(6)]]]],M(96)[M(48)[M(24)[M(12)[46(6),195(6)],112(12)],M(24)[40(12),M(12)[98(6),175(6)]]],M(48)[M(24)[182(12),M(12)[158(6),246(6)]],M(24)[M(12)[103(6),30(6)],8(12)]]]],M(192)[M(96)[M(48)[M(24)[193(12),85(12)],M(24)[141(12),143(12)]],M(48)[M(24)[11(12),4(12)],M(24)[12(12),132(12)]]],M(96)[M(48)[M(24)[47(12),M(12)[225(6),252(6)]],M(24)[M(12)[159(6),178(6)],242(12)]],M(48)[M(24)[M(12)[108(6),138(6)],M(12)[96(6),209(6)]],M(24)[M(12)[120(6),208(6)],118(12)]]]]]]]]
   ```

   the error happened on `swap curr=TNone:(TNone:(T256:(,),T210:(,)),T224:(,)) with first_same_weight=T224:(,)`

   This is caused by the wrong order of `add_one()` in this code

   ```py
   for _ in range(2):
      self.__list.add_one(curr)
       curr = curr.parent
   ```

   The parent is added to the larger weight group before the child, which then caused in the next update, the parent's "first_same_weight" node is its child, which is not allowed to swap with.
   The correct order should be

   ```py
   for node in curr.parent.parent, curr.parent:
       self.__list.add_one(node)
   curr = curr.parent.parent
   ```

7. During fixing 6, I tried to visualize the tree. ChatGPT told me to use `networkx` with `matplotlib`. But I found that `pydot` could do this and tried it. The result is not meaningful: but I see that the `label` of `pydot.Node` cannot be 'node' or 'Node' (reserved word)
8. During fixing 6, in `AdaptiveHuffmanTree.serialize_node()` I noticed that black 23.11.0 does not support py3.12 `""` inside `""`, formally "quote reuse", "PEP 701: Syntactic formalization of f-strings". See details in <https://docs.python.org/3/whatsnew/3.12.html#pep-701-syntactic-formalization-of-f-strings>.
9. Performance issue. After the encoding / decoding, getting the `Bits.__repr__()` was very slow. This is due to the binary to decimal conversion. If we use only binary form for the long strings, the stdout print speed is more-than-10-times faster. But encoding 1-10k bytes takes longer than I expected, on my local machine.
10. Another problem after fixing 6 is that we should never point the updating node to the dummy root, otherwise we cannot encode bytes with same first two bytes, like `aa...`, `xx...`
