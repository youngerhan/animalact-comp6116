# Action Class Reference

The vocabulary contains 119 action classes. Vocabulary version: `1.0.0`. Split version: `animalact-70-15-15-v1`. Global class IDs remain stable across compatible dataset revisions.

Read action names from `annotations[].actions[].action`. Names are case-sensitive. One video can contain multiple actions, so per-class video counts do not sum to the dataset size. An interval is one action record. Groups connect videos with identical file hashes or sampled frames; they do not establish complete source-recording independence.

See [MAIN_EVALUATION.md](MAIN_EVALUATION.md) for the evaluation protocol. Machine-readable counts are available in [action_classes.json](action_classes.json) and [splits/class_statistics.csv](splits/class_statistics.csv).

| ID | Action | Intervals | Videos | Train videos | Valid videos | Test videos | Groups | Main evaluation |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | Abseiling | 1 | 1 | 1 | 0 | 0 | 1 | No |
| 1 | Attacking | 48 | 42 | 30 | 6 | 6 | 38 | Yes |
| 2 | Attending | 501 | 355 | 248 | 53 | 54 | 316 | Yes |
| 3 | Barking | 17 | 15 | 11 | 2 | 2 | 12 | No |
| 4 | Being Carried | 1 | 1 | 1 | 0 | 0 | 1 | No |
| 5 | Being Carried In Mouth | 5 | 4 | 2 | 1 | 1 | 4 | No |
| 6 | Being Dragged | 3 | 3 | 1 | 1 | 1 | 3 | No |
| 7 | Being Eaten | 23 | 23 | 16 | 3 | 4 | 23 | Yes |
| 8 | Biting | 60 | 53 | 37 | 8 | 8 | 51 | Yes |
| 9 | Building nest | 14 | 10 | 7 | 1 | 2 | 10 | No |
| 10 | Calling | 3 | 3 | 1 | 1 | 1 | 3 | No |
| 11 | Camouflaging | 1 | 1 | 1 | 0 | 0 | 1 | No |
| 12 | Carrying | 5 | 5 | 3 | 1 | 1 | 5 | No |
| 13 | Carrying In Mouth | 29 | 28 | 20 | 4 | 4 | 28 | Yes |
| 14 | Chasing | 7 | 6 | 4 | 1 | 1 | 6 | No |
| 15 | Chirping | 518 | 277 | 193 | 42 | 42 | 237 | Yes |
| 16 | Climbing | 36 | 34 | 24 | 5 | 5 | 34 | Yes |
| 17 | Coiling | 1 | 1 | 1 | 0 | 0 | 1 | No |
| 18 | Competing For Dominance | 4 | 4 | 2 | 1 | 1 | 4 | No |
| 19 | Dancing | 9 | 6 | 4 | 1 | 1 | 6 | No |
| 20 | Dancing On Water | 1 | 1 | 1 | 0 | 0 | 1 | No |
| 21 | Dead | 8 | 8 | 6 | 1 | 1 | 8 | No |
| 22 | Defecating | 5 | 5 | 3 | 1 | 1 | 4 | No |
| 23 | Defensive Rearing | 2 | 1 | 1 | 0 | 0 | 1 | No |
| 24 | Detaching As A Parasite | 1 | 1 | 1 | 0 | 0 | 1 | No |
| 25 | Digging | 18 | 17 | 12 | 2 | 3 | 17 | No |
| 26 | Displaying Defensive Pose | 56 | 23 | 16 | 3 | 4 | 20 | Yes |
| 27 | Disturbing Another Animal | 78 | 43 | 30 | 6 | 7 | 38 | Yes |
| 28 | Diving | 11 | 10 | 7 | 2 | 1 | 10 | No |
| 29 | Doing A Back Kick | 4 | 4 | 2 | 1 | 1 | 4 | No |
| 30 | Doing A Backward Tilt | 3 | 3 | 1 | 1 | 1 | 3 | No |
| 31 | Doing A Chin Dip | 2 | 1 | 1 | 0 | 0 | 1 | No |
| 32 | Doing A Face Dip | 75 | 34 | 24 | 5 | 5 | 28 | Yes |
| 33 | Doing A Neck Raise | 12 | 8 | 6 | 1 | 1 | 6 | No |
| 34 | Doing A Side Tilt | 1 | 1 | 1 | 0 | 0 | 1 | No |
| 35 | Doing somersault | 2 | 1 | 1 | 0 | 0 | 1 | No |
| 36 | Drifting | 6 | 6 | 4 | 1 | 1 | 6 | No |
| 37 | Drinking | 112 | 60 | 42 | 9 | 9 | 54 | Yes |
| 38 | Dying | 1 | 1 | 1 | 0 | 0 | 1 | No |
| 39 | Eating | 400 | 319 | 223 | 48 | 48 | 307 | Yes |
| 40 | Entering its nest | 9 | 9 | 6 | 2 | 1 | 9 | No |
| 41 | Escaping | 1 | 1 | 1 | 0 | 0 | 1 | No |
| 42 | Exiting Cocoon | 2 | 2 | 2 | 0 | 0 | 2 | No |
| 43 | Exiting Nest | 4 | 4 | 2 | 1 | 1 | 4 | No |
| 44 | Exploring | 59 | 57 | 40 | 9 | 8 | 57 | Yes |
| 45 | Falling | 19 | 19 | 13 | 3 | 3 | 16 | No |
| 46 | Fighting | 85 | 53 | 37 | 8 | 8 | 51 | Yes |
| 47 | Flapping | 102 | 62 | 44 | 9 | 9 | 57 | Yes |
| 48 | Flapping Tail | 17 | 15 | 11 | 2 | 2 | 13 | No |
| 49 | Fleeing | 13 | 12 | 8 | 2 | 2 | 11 | No |
| 50 | Flying | 182 | 144 | 99 | 24 | 21 | 137 | Yes |
| 51 | Giving Birth | 1 | 1 | 1 | 0 | 0 | 1 | No |
| 52 | Gliding | 3 | 3 | 1 | 1 | 1 | 3 | No |
| 53 | Grooming | 34 | 34 | 24 | 5 | 5 | 33 | Yes |
| 54 | Hanging | 8 | 8 | 6 | 1 | 1 | 8 | No |
| 55 | Having A Flehmen Response | 3 | 2 | 2 | 0 | 0 | 2 | No |
| 56 | Hissing | 14 | 10 | 7 | 2 | 1 | 9 | No |
| 57 | Holding Hands | 1 | 1 | 1 | 0 | 0 | 1 | No |
| 58 | Hopping | 4 | 3 | 1 | 1 | 1 | 3 | No |
| 59 | Hugging | 23 | 23 | 16 | 4 | 3 | 21 | Yes |
| 60 | Immobilized | 2 | 2 | 2 | 0 | 0 | 2 | No |
| 61 | Jumping | 117 | 94 | 66 | 14 | 14 | 92 | Yes |
| 62 | Keeping still | 1108 | 788 | 552 | 119 | 117 | 730 | Yes |
| 63 | Landing | 90 | 72 | 50 | 12 | 10 | 68 | Yes |
| 64 | Laying Eggs | 3 | 3 | 1 | 1 | 1 | 3 | No |
| 65 | Leaning | 3 | 3 | 1 | 1 | 1 | 3 | No |
| 66 | Licking | 4 | 4 | 2 | 1 | 1 | 4 | No |
| 67 | Lying Down | 9 | 8 | 6 | 1 | 1 | 7 | No |
| 68 | Lying On Top | 13 | 13 | 9 | 2 | 2 | 13 | No |
| 69 | Manipulating Object | 13 | 13 | 9 | 2 | 2 | 12 | No |
| 70 | Moving | 462 | 389 | 272 | 58 | 59 | 382 | Yes |
| 71 | Pecking | 21 | 12 | 8 | 2 | 2 | 11 | No |
| 72 | Performing Sexual Display | 3 | 3 | 1 | 1 | 1 | 3 | No |
| 73 | Performing copulatory mounting | 7 | 7 | 5 | 1 | 1 | 7 | No |
| 74 | Playing | 9 | 9 | 6 | 2 | 1 | 9 | No |
| 75 | Playing Dead | 2 | 1 | 1 | 0 | 0 | 1 | No |
| 76 | Preening | 159 | 109 | 76 | 17 | 16 | 94 | Yes |
| 77 | Preying | 1 | 1 | 1 | 0 | 0 | 1 | No |
| 78 | Puffing its throat | 1 | 1 | 1 | 0 | 0 | 1 | No |
| 79 | Rattling | 5 | 5 | 3 | 1 | 1 | 5 | No |
| 80 | Resting | 2 | 2 | 2 | 0 | 0 | 2 | No |
| 81 | Retaliating | 13 | 13 | 9 | 2 | 2 | 13 | No |
| 82 | Retreating | 19 | 17 | 12 | 3 | 2 | 12 | No |
| 83 | Rolling | 3 | 3 | 1 | 1 | 1 | 3 | No |
| 84 | Rubbing its head | 12 | 9 | 6 | 2 | 1 | 8 | No |
| 85 | Running | 113 | 94 | 66 | 14 | 14 | 92 | Yes |
| 86 | Running On Water | 4 | 4 | 2 | 1 | 1 | 4 | No |
| 87 | Sensing | 1084 | 733 | 513 | 110 | 110 | 676 | Yes |
| 88 | Shaking | 46 | 33 | 23 | 5 | 5 | 30 | Yes |
| 89 | Shaking Head | 184 | 113 | 79 | 17 | 17 | 94 | Yes |
| 90 | Sharing Food | 24 | 18 | 12 | 3 | 3 | 18 | No |
| 91 | Showing Affection | 3 | 3 | 1 | 1 | 1 | 3 | No |
| 92 | Sinking | 1 | 1 | 1 | 0 | 0 | 1 | No |
| 93 | Sitting | 20 | 19 | 13 | 3 | 3 | 16 | No |
| 94 | Sleeping | 1 | 1 | 1 | 0 | 0 | 1 | No |
| 95 | Sleeping in its nest | 6 | 3 | 1 | 1 | 1 | 3 | No |
| 96 | Spitting Venom | 46 | 41 | 29 | 6 | 6 | 40 | Yes |
| 97 | Spreading Wings | 26 | 18 | 12 | 3 | 3 | 14 | No |
| 98 | Squatting | 2 | 2 | 2 | 0 | 0 | 2 | No |
| 99 | Standing | 17 | 15 | 11 | 2 | 2 | 14 | No |
| 100 | Standing in alert | 5 | 5 | 3 | 1 | 1 | 5 | No |
| 101 | Startled | 20 | 15 | 11 | 2 | 2 | 13 | No |
| 102 | Stinging | 1 | 1 | 1 | 0 | 0 | 1 | No |
| 103 | Struggling | 20 | 17 | 12 | 3 | 2 | 17 | No |
| 104 | Surfacing | 2 | 2 | 2 | 0 | 0 | 2 | No |
| 105 | Swaying | 1 | 1 | 1 | 0 | 0 | 1 | No |
| 106 | Swimming | 65 | 61 | 42 | 10 | 9 | 59 | Yes |
| 107 | Swimming In Circles | 1 | 1 | 1 | 0 | 0 | 1 | No |
| 108 | Tail Swishing | 2 | 2 | 2 | 0 | 0 | 2 | No |
| 109 | Trapped | 4 | 4 | 2 | 1 | 1 | 4 | No |
| 110 | Turning Around | 40 | 33 | 22 | 6 | 5 | 32 | Yes |
| 111 | Undergoing Chrysalis | 3 | 3 | 1 | 1 | 1 | 3 | No |
| 112 | Unmounting | 1 | 1 | 1 | 0 | 0 | 1 | No |
| 113 | Urinating | 7 | 4 | 2 | 1 | 1 | 3 | No |
| 114 | Walking | 568 | 448 | 313 | 68 | 67 | 419 | Yes |
| 115 | Washing | 43 | 22 | 16 | 3 | 3 | 19 | No |
| 116 | Wrapping Itself Around Prey | 1 | 1 | 1 | 0 | 0 | 1 | No |
| 117 | Wrapping Prey | 1 | 1 | 1 | 0 | 0 | 1 | No |
| 118 | Yawning | 12 | 12 | 8 | 2 | 2 | 12 | No |
