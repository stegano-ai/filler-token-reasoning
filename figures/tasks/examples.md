## mod sum (`mod2`, 10 steps)

```
You will get a list of 10 numbers, each 0 or 1. Is the count of 1s in the list even or odd?

For example, [1, 0, 1] contains two 1s, so the count is even and the answer is 0.

The list: [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]

Answer with a single digit after ####: 0 if the count of 1s is even, 1 if it is odd.
```

reasoning states (one per dot): `[0, 0, 1, 1, 1, 1, 1, 1, 0, 0]`  
answer: `0`

## A5 (`a5`, 10 steps)

```
Five cups sit at positions 0-4; a ball starts under the cup at position 0. You will get a list of moves, each 0 or 1. Apply them in order. 0 = Rotate: every cup moves from position p to (p+1) mod 5. 1 = Double-Swap: the cups at positions 0 and 1 swap, and the cups at positions 3 and 4 swap.

For example, for moves [1, 0]: the Double-Swap moves the ball from position 0 to position 1, then the Rotate moves it from position 1 to position 2, so the answer is 2.

Moves: [0, 0, 0, 0, 0, 0, 1, 0, 1, 1]

Where is the ball at the end? Answer with a single digit from 0 to 4 after ####.
```

reasoning states (one per dot): `[1, 2, 3, 4, 0, 1, 0, 1, 0, 1]`  
answer: `1`

## GL(3,2) (`gl32`, 10 steps)

```
Three lamps sit in a ring: lamp A, lamp B, lamp C. Only lamp A is on at the start. You will get a list of moves, each 0 or 1. Apply them in order. 0 = Shift: every lamp passes its on/off state to the next lamp (A's state moves to B, B's to C, C's to A). 1 = Couple: if lamp A is on, lamp B toggles; if lamp A is off, nothing happens.

Encode a lamp pattern as the number A + 2*B + 4*C, counting an on lamp as 1. For example, for moves [1, 0]: the Couple turns lamp B on (lamp A is on), then the Shift passes each state along the ring, leaving lamps B and C on, so the answer is 6.

Moves: [0, 0, 1, 1, 1, 0, 0, 1, 0, 0]

What is the final pattern number? Answer with a single digit from 1 to 7 after ####.
```

reasoning states (one per dot): `[2, 4, 4, 4, 4, 1, 2, 2, 4, 1]`  
answer: `1`

## modular arithmetic (`arith8mod`, 3 steps)

```
You will receive a starting value followed by 3 operations using +, -, *. Evaluate left-to-right with no operator precedence, reducing modulo 8 after every operation, so every intermediate result is an integer in [0, 7]. Use the 3 intermediate results as your reasoning, and output the final result. For example, with modulus 7, given '5 + 4 * 3 - 2' (3 steps): step 1: 5+4=9=2 mod 7, step 2: 2*3=6=6 mod 7, step 3: 6-2=4=4 mod 7. Reasoning is [2, 6, 4], output is [4].

Your input: 1 + 6 - 2 + 2

Answer with a single digit from 0 to 7 after ####: the final result.
```

reasoning states (one per dot): `[7, 5, 7]`  
answer: `7`

## ProntoQA (`chain`, 4 steps)

```
You will receive a list of numbered premises followed by a query of the form "True or false: ...". Exactly one premise is a base fact placing an entity in a category (e.g. "Alex is a wumpus."); the rest are rules of the form "Every A is a B.". Each category has at most one rule leading out of it, so from the entity's starting category there is exactly one chain of 4 rules to follow; the other rules form separate chains that the entity's chain never touches. Follow the entity's chain to its final category. The queried category is the final category of one of the chains; the answer is 1 if it is the final category of the entity's chain, 0 if it belongs to a different chain. Emit your reasoning as premise numbers in the order you use them, each written in binary (most significant bit first, no leading zeros) and separated by the delimiter value 2: first the number of the base-fact premise, then the number of each rule you follow, for 4 rules in total. Conclude your reasoning with one more delimiter 2 followed by the answer: 1 if the queried category is the entity's final category, 0 if it is not. Then output that same single value. For example, if the base fact is premise 5 and the first rule you follow is premise 2, your reasoning begins [1, 0, 1, 2, 1, 0] (premise 5 = binary 101, delimiter 2, premise 2 = binary 10), continuing with the remaining rules and ending with a delimiter 2 and the answer.

0. Every tevpor is a tralathpant.
1. Every woreepor is a tevpor.
2. Every dromumper is a woreepor.
3. Every grumalpee is a quiminpor.
4. Every grumathpor is a grumalpee.
5. Every pogeepist is a dromumper.
6. Every quiminpor is a sibathpant.
7. Every sibathpant is a dromorgit.
8. Wren is a pogeepist.

True or false: Wren is a tralathpant.

Answer with a single digit after ####: 1 if the queried statement is true, 0 if it is false.
```

reasoning states (one per dot): `[1, 0, 0, 0, 2, 1, 0, 1, 2, 1, 0, 2, 1, 2, 0, 2, 1]`  
answer: `1`

## HornQA (`horn`, 8 steps)

```
You will receive numbered premises followed by a query "? E : X". A premise "E : A" is a base fact: entity E is in category A. A premise "A B > C" is a rule: anything that is in BOTH A and B is also in C, so the rule fires for an entity only once both body categories A and B are established for it, and then establishes C. Categories are shared between entities, so no single category says which entity a rule belongs to — only the PAIR of body categories is specific to one entity, and every rule fires for exactly one of them. The queried entity has exactly two base facts; starting from those two categories, fire rules as they become applicable, 8 rules in total, down to the entity's final category. Exactly one rule is applicable at each step: the one pairing the category you established last with some category established earlier, which may be arbitrarily far back — always check both body categories against everything established so far. The queried category may instead be another entity's final category; the answer is whether it is derivable FOR THE QUERIED ENTITY. Emit your reasoning as premise numbers in use order, each in binary (most significant bit first, no leading zeros), separated by the delimiter value 2: first the queried entity's two base-fact premises in increasing order, then each rule you fire in derivation order, 8 rules in total. Conclude with one more delimiter 2 and the answer: 1 if the queried category is the entity's final category, 0 if it is not. Then output that same single value. For example, base-fact premises 2 and 5 begin the reasoning [1, 0, 2, 1, 0, 1] (premise 2 = binary 10, delimiter 2, premise 5 = binary 101).

0. pie earth > ruby
1. Sam : star
2. star earth > band
3. band earth > pie
4. month store > flute
5. Alex : pine
6. band pine > sea
7. flower saw > bean
8. mug mouse > band
9. pie bean > saw
10. star dress > flute
11. mouse carrot > angle
12. mouse flute > ruby
13. flower bean > roll
14. window pie > carrot
15. angle band > flower
16. store window > month
17. store carrot > mouse
18. Alex : mug
19. Stella : dress
20. pie ruby > month
21. pine bean > pie
22. flute star > earth
23. flower ruby > earth
24. mouse angle > mug
25. pine saw > window
26. Stella : window
27. pie angle > room
28. Fae : carrot
29. window flute > flower
30. carrot window > star
31. dress flower > mouse
32. Sam : dress
33. star window > angle
34. pie month > pine
35. Fae : store
36. flower earth > plum
37. mug pine > bean
38. flower mug > saw
39. dress window > store

? Sam : sea

Answer with a single digit after ####: 1 if the queried statement is true, 0 if it is false.
```

reasoning states (one per dot): `[1, 2, 1, 0, 0, 0, 0, 0, 2, 1, 0, 1, 0, 2, 1, 0, 1, 1, 0, 2, 1, 0, 2, 1, 1, 2, 0, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0, 0, 1, 0, 2, 1, 1, 0, 2, 1]`  
answer: `1`

## function composition (`fc`, 3 steps)

```
You will receive a list of integers: a start value, then 3 lookup tables of 8 entries each, written one after another. Each table maps a value v (from 0 to 7) to its entry at index v. Starting from the start value, apply table 1, then table 2, and so on, feeding each result into the next table. Track the value after each table as your reasoning, then output those values. For example, with 2 tables of 4 entries each, the input [1, 3, 0, 2, 1, 1, 3, 0, 2] means start value 1, table 1 = [3, 0, 2, 1], table 2 = [1, 3, 0, 2]: table 1 maps 1 to 0, then table 2 maps 0 to 1, so reasoning is [0, 1], output is [0, 1].

[4, 6, 1, 3, 0, 2, 5, 4, 7, 4, 6, 5, 7, 1, 0, 3, 2, 0, 4, 2, 3, 7, 6, 1, 5]

Answer with a single digit after ####: the final value after applying all the tables.
```

reasoning states (one per dot): `[2, 5, 6]`  
answer: `6`
