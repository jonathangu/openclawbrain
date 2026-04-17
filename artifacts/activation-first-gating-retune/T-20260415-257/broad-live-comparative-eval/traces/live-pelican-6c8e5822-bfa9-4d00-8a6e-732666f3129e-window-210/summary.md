# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-210`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9de933020cc2b0d03a4b1b4f5bf51c7bca0c4ae8de78af7a2cf6d4b86ac284d4`
- fixture hash: `sha256-2e5835aa933cf2df6faf2714837c2953d1866a5094413604d0ec3e648b5257c4`
- score hash: `sha256-c22a517b1fd30e465e3e2f45c7cef60d3aba376ed0446cd503802ae922708445`
- bundle hash: `sha256-b5ae0ac0073d81ace619aed53ba0d99b51250251c76d40802a255a24e573a56d`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/4
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-13242c9a12ffb4d788c2f14891b978d17c5b819a44b8fb4dd405e1c1b50322e8 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3c798f6721a699b3175ea283a7c4cca5cb6d48a824c40447859d51fa6253c24d |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b1b3d1f6eea70bcba0c8bd04a3a9e03157eebe9917801a78edf34760c7f717f7 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-06fa787f43e3e110ee60628e1c42944123f1ba57297b9687d3c5b0160b7f3e9b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-81538908 | sha256-6acb6425ff1a3759a74c53935165d71e3c76c5334f8a16a06786f81a69dac8b1 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-81538908 | sha256-47c0f67f5fcc8c7f9d3ca5b19e5a47df844d9068837f2336957982a10e28b20d |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-36887a3b | sha256-019a94d5115ddc8040d18d2f933f8585896015a09af2192f9dc97f9860322994 |
