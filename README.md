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

6. Another edge case with

   ```text
   T[ROOT]None:(TNone:(TNone:(TNone:(TNone:(TNone:(TNone:(TNone:(T214:(,),T80:(,)),TNone:(T81:(,),T231:(,))),TNone:(T27:(,),TNone:(T50:(,),T66:(,)))),TNone:(TNone:(TNone:(T48:(,),T69:(,)),T241:(,)),TNone:(T203:(,),TNone:(T148:(,),T160:(,))))),TNone:(TNone:(TNone:(T114:(,),T243:(,)),TNone:(TNone:(T229:(,),T255:(,)),T109:(,))),TNone:(TNone:(T136:(,),T71:(,)),TNone:(T18:(,),T25:(,))))),TNone:(TNone:(TNone:(TNone:(T196:(,),T122:(,)),TNone:(T106:(,),T32:(,))),TNone:(TNone:(T52:(,),T218:(,)),TNone:(T164:(,),TNone:(T21:(,),T226:(,))))),TNone:(TNone:(TNone:(T212:(,),T253:(,)),TNone:(TNone:(T87:(,),T222:(,)),T89:(,))),TNone:(TNone:(T230:(,),TNone:(T168:(,),TNone:(T73:(,),T146:(,)))),TNone:(T126:(,),T128:(,)))))),TNone:(TNone:(TNone:(TNone:(TNone:(TNone:(T6:(,),T34:(,)),T165:(,)),TNone:(TNone:(T219:(,),T117:(,)),TNone:(T10:(,),T213:(,)))),TNone:(TNone:(TNone:(T0:(,),T133:(,)),TNone:(T197:(,),T202:(,))),TNone:(TNone:(T236:(,),T110:(,)),T254:(,)))),TNone:(TNone:(TNone:(TNone:(T154:(,),T233:(,)),TNone:(T56:(,),T115:(,))),TNone:(TNone:(T139:(,),T124:(,)),TNone:(T130:(,),T161:(,)))),TNone:(TNone:(TNone:(T91:(,),T16:(,)),TNone:(T104:(,),T248:(,))),TNone:(TNone:(T250:(,),T5:(,)),TNone:(T249:(,),T125:(,)))))),TNone:(TNone:(TNone:(TNone:(TNone:(T174:(,),T228:(,)),TNone:(T107:(,),T76:(,))),TNone:(TNone:(TNone:(T7:(,),T93:(,)),T162:(,)),TNone:(TNone:(T184:(,),T185:(,)),T77:(,)))),TNone:(TNone:(TNone:(T156:(,),T140:(,)),TNone:(TNone:(T179:(,),T101:(,)),T199:(,))),TNone:(TNone:(T36:(,),T186:(,)),TNone:(T191:(,),TNone:(T60:(,),T86:(,)))))),TNone:(TNone:(TNone:(TNone:(T181:(,),T119:(,)),TNone:(T65:(,),T152:(,))),TNone:(TNone:(T166:(,),T123:(,)),TNone:(T79:(,),T201:(,)))),TNone:(TNone:(TNone:(T54:(,),T99:(,)),T14:(,)),TNone:(TNone:(T23:(,),T198:(,)),TNone:(T63:(,),T13:(,)))))))),TNone:(TNone:(TNone:(TNone:(TNone:(TNone:(TNone:(T20:(,),T157:(,)),TNone:(T92:(,),T227:(,))),TNone:(TNone:(T220:(,),T215:(,)),TNone:(T187:(,),T43:(,)))),TNone:(TNone:(TNone:(T90:(,),T83:(,)),TNone:(T221:(,),T134:(,))),TNone:(TNone:(T38:(,),T49:(,)),TNone:(T17:(,),T97:(,))))),TNone:(TNone:(TNone:(T1:(,),TNone:(T51:(,),T113:(,))),TNone:(TNone:(T145:(,),T216:(,)),TNone:(T239:(,),T68:(,)))),TNone:(TNone:(TNone:(T163:(,),T58:(,)),TNone:(T142:(,),T240:(,))),TNone:(TNone:(T53:(,),T200:(,)),TNone:(T173:(,),T238:(,)))))),TNone:(TNone:(TNone:(TNone:(T151:(,),TNone:(T131:(,),TNone:(T45:(,),T70:(,)))),TNone:(TNone:(T88:(,),T82:(,)),TNone:(T204:(,),T235:(,)))),TNone:(TNone:(TNone:(T59:(,),T247:(,)),TNone:(T64:(,),T62:(,))),TNone:(TNone:(T61:(,),TNone:(T189:(,),T150:(,))),TNone:(T94:(,),TNone:(T149:(,),T24:(,)))))),TNone:(TNone:(TNone:(TNone:(TNone:(T171:(,),T223:(,)),T2:(,)),TNone:(T245:(,),TNone:(T144:(,),T55:(,)))),TNone:(TNone:(TNone:(T135:(,),T205:(,)),T147:(,)),TNone:(TNone:(T67:(,),TNone:(TNone:(T256:(,),T224:(,)),T39:(,))),TNone:(T33:(,),T35:(,))))),TNone:(TNone:(TNone:(T29:(,),T22:(,)),TNone:(TNone:(T167:(,),T232:(,)),T180:(,))),TNone:(TNone:(T190:(,),TNone:(T129:(,),T170:(,))),TNone:(T95:(,),T244:(,))))))),TNone:(TNone:(TNone:(TNone:(TNone:(TNone:(TNone:(T15:(,),T26:(,)),T44:(,)),TNone:(T153:(,),T74:(,))),TNone:(TNone:(T137:(,),T102:(,)),TNone:(T28:(,),T121:(,)))),TNone:(TNone:(TNone:(T78:(,),T72:(,)),TNone:(T251:(,),T127:(,))),TNone:(TNone:(T41:(,),T116:(,)),TNone:(T37:(,),T111:(,))))),TNone:(TNone:(TNone:(TNone:(T155:(,),T3:(,)),TNone:(T188:(,),T105:(,))),TNone:(TNone:(T57:(,),T234:(,)),TNone:(T84:(,),T100:(,)))),TNone:(TNone:(TNone:(T169:(,),T192:(,)),TNone:(T31:(,),T75:(,))),TNone:(TNone:(T183:(,),T194:(,)),TNone:(T9:(,),T42:(,)))))),TNone:(TNone:(TNone:(TNone:(TNone:(T237:(,),T172:(,)),TNone:(T207:(,),T217:(,))),TNone:(TNone:(T211:(,),T176:(,)),TNone:(T206:(,),TNone:(T177:(,),T19:(,))))),TNone:(TNone:(TNone:(TNone:(T46:(,),T195:(,)),T112:(,)),TNone:(T40:(,),TNone:(T98:(,),T175:(,)))),TNone:(TNone:(T182:(,),TNone:(T158:(,),T246:(,))),TNone:(TNone:(T103:(,),T30:(,)),T8:(,))))),TNone:(TNone:(TNone:(TNone:(T193:(,),T85:(,)),TNone:(T141:(,),T143:(,))),TNone:(TNone:(T11:(,),T4:(,)),TNone:(T12:(,),T132:(,)))),TNone:(TNone:(TNone:(T47:(,),TNone:(T225:(,),T252:(,))),TNone:(TNone:(T159:(,),T178:(,)),T242:(,))),TNone:(TNone:(TNone:(T108:(,),T138:(,)),TNone:(T96:(,),T209:(,))),TNone:(TNone:(T120:(,),T208:(,)),T118:(,)))))))))
   ```

   happened on `swap curr=TNone:(TNone:(T256:(,),T210:(,)),T224:(,)) with first_same_weight=T224:(,)`
