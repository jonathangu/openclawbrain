# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-500bb42a51fe35739e28b1f6be3d9fe7ff92c6a8eeb2f053f3018ae2eba88584`
- fixture hash: `sha256-f69dca5c27c722f582ac3debb2e25adae4c35c5bd6a4749aa476e37eee07c7bc`
- score hash: `sha256-4965f5bd6c2cc30455aa24e76a066e84f239fb89658a17425dbe18474238e4df`
- bundle hash: `sha256-d5403423803953095a343f2f769cd44f0ff1041f3be72a6226be8cd78c571baa`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 0 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1a93c94aa4cb26ac67e3ba4bdee5fc22bb0276c3da7ff11089c43e42405c272c |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-b2b3abe16d3357272fdcfb016017e46e8b63f2febb27a90f14555c0e36770169 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-dda28a28f0638b61601e667004a5e04891749d03294534fe2f3df19498baa7e1 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-61d0c5066b8bb270729c6dd374cfc07f54c02dae829d3f726a2d159b5a032727 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-b56bf5bc | sha256-7280deb3449a92fed9acb4c9a252f7b517d2ee420edb5052050b814c13881c52 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-b56bf5bc | sha256-94ccd9ace59794ec0212b88d775683e37b6c4c1d2bc1b732616cec5a18e9d66c |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-f78dd5b5 | sha256-7c9707b82865f5df560022d811c6a7f73786fad91bf5f214a3a4431f16a1229c |
