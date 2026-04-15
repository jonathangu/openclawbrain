# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-032`
- winner mode: `graph_prior_only`
- trace hash: `sha256-92a53a83b75391e6ea2e19694e75cc46987c1fd7f2482c72c3850eb3ee758d5b`
- fixture hash: `sha256-a7a70c06edd57e7fef42061ce44261270b10f99213ced50cea189f13c03e8e7a`
- score hash: `sha256-bfe50bb2d3d4922dd499aacfb0b8378175958e96cae5a70a01dbdabd76758924`
- bundle hash: `sha256-575d5f789fa5001054e7fdb4c75786824584a7e8012f303e5a29272181070809`

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
- phrase hits: 0/12
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-467d90a6c748c6c78cf3c7ceb933156139020979bf5f7ad7e3a8103479da429a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-01539069889e8365f6011546fa7bba0cfcc86246621c89952a58f6df7fb0e281 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-11b7bfcbbd719150740041e0d6b1a855801775d7ebc9c7125a4fab7d0aecf7b5 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-6bf4392fd0dffe90bf68c4f7886a85b6d0fb93d5f16d72dcba2f9ba0ba7c9734 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d96e797c | sha256-2a1e6d5910c1f9329e18281d439e1e450c239e131bdd66b481947ce7186eff90 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d96e797c | sha256-b938d9a8306054ab9abb9dec7008a2553499db759a23b5e36b1afc685bc2a497 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-d96e797c | sha256-2a1e6d5910c1f9329e18281d439e1e450c239e131bdd66b481947ce7186eff90 |
