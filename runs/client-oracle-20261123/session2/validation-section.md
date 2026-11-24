## Twenty-fight recorded-seed validation

**20/20** reproduced the recorded winner and the entire terminal list of survivors, in order, with exact attack and HP. Each used a distinct dev match and unaltered original Battle bytes. The selection was fixed by SHA-256 before play and excluded known engine misses; this checks the client as a reference, independently of the later failing-fight diagnoses. All 20 file hashes appear in the local-fulfillment receipt. Terminal observations were read visually, written to a separate ledger, and then compared with Outcome and ResultingBoard. [Frame readings](session2/terminal-manual.tsv) · [Per-fight comparison](session2/validation-results.json)

Median time from the verified rewatch click to the first blocked watched acknowledgment was **37.5 s** (range 21.6–50.9). Including ordinary reload and menu navigation, median wall time was 68.8 s per fixture. This is seed-exact endpoint replay on the tested v0.48 fights. It establishes a real-client reference for intermediate-frame inspection; recognition of each intermediate frame is a separate measurement.

| Fight | Winner on screen | Survivors read directly, screen left to right | Seconds |
|---|---|---|---:|
| [validation-01](session2/captures/validation-01/frame-0052.jpg) | L | Rabbit 3/5, Snail 3/5 | 37.4 |
| [validation-02](session2/captures/validation-02/frame-0061.jpg) | L | Skunk 4/3 | 45.6 |
| [validation-03](session2/captures/validation-03/frame-0057.jpg) | L | Turkey 3/4, Bus 8/4, Crab 16/11 | 42.5 |
| [validation-04](session2/captures/validation-04/frame-0027.jpg) | R | Pigeon 9/1, Beaver 4/3, Beaver 4/3 | 21.6 |
| [validation-05](session2/captures/validation-05/frame-0040.jpg) | L | Parrot 9/2 | 29.7 |
| [validation-06](session2/captures/validation-06/frame-0055.jpg) | R | Horse 6/8 | 41.8 |
| [validation-07](session2/captures/validation-07/frame-0051.jpg) | L | Shark 27/36, Rooster 10/11, Swan 28/6 | 39.5 |
| [validation-08](session2/captures/validation-08/frame-0045.jpg) | L | Snail 2/1 | 32.7 |
| [validation-09](session2/captures/validation-09/frame-0050.jpg) | R | Fish 5/1 | 37.4 |
| [validation-10](session2/captures/validation-10/frame-0051.jpg) | L | Snail 4/4 | 37.6 |
| [validation-11](session2/captures/validation-11/frame-0074.jpg) | R | Wolverine 5/2 | 43.0 |
| [validation-12](session2/captures/validation-12/frame-0059.jpg) | R | Wolverine 7/13 | 34.0 |
| [validation-13](session2/captures/validation-13/frame-0062.jpg) | L | Ox 1/6, Kangaroo 3/3, Ox 3/7 | 34.8 |
| [validation-14](session2/captures/validation-14/frame-0090.jpg) | D | empty | 50.9 |
| [validation-15](session2/captures/validation-15/frame-0065.jpg) | R | Ox 47/9 | 39.1 |
| [validation-16](session2/captures/validation-16/frame-0060.jpg) | L | Wolverine 7/2, Blowfish 11/7 | 38.4 |
| [validation-17](session2/captures/validation-17/frame-0044.jpg) | L | Crab 4/3 | 24.9 |
| [validation-18](session2/captures/validation-18/frame-0078.jpg) | R | Shark 26/26 | 45.1 |
| [validation-19](session2/captures/validation-19/frame-0059.jpg) | R | Dog 11/1 | 35.0 |
| [validation-20](session2/captures/validation-20/frame-0043.jpg) | R | Leopard 29/8, Dragon 9/13 | 30.5 |
