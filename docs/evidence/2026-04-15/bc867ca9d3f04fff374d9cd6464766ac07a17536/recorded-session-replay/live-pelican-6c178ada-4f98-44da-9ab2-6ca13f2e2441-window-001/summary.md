# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c178ada-4f98-44da-9ab2-6ca13f2e2441-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5c1146574706ec395e6f5011fe3bdf3e510b31ef69670a55eff27bc156061d1f`
- fixture hash: `sha256-b0b87869202da9099b109d7a7b86f16484e8b3960b663b22dcb9b0c0fd925784`
- score hash: `sha256-3aea31178f99deab30f4c0e24af2b1d2911d7fd6d30059497c3431da65d0cc21`
- bundle hash: `sha256-f80c9168c772fb9eff744d91a2bb71c7f6a5a1543be8b209cd011b60e5ad6980`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4ee8a7a4748f6af35d73940b990960d0c8506d722d1756ec1464f9fd52079a6e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-61619980ff5b37d1cf998301721b29f8141f239602b04e6f95e0af7c19b40126 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3a26bf0c76cd01aa7a8e910a1d7af483d0cb2c0dc478e5734928834af24b92e8 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-2dd91101097af3c0b18473e7621f7d8205d78162773585ff6e731214cb52fef0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a5e9bfbd | sha256-9626ac02c8ab78935c9e67002676652497c1e6c20c8e9b20de8b9501209f14ae |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a5e9bfbd | sha256-58a57e72b5833d5de116c63d90b5eecb7ecb4f70a383d8f502bd2518f74435d2 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-a5e9bfbd | sha256-9626ac02c8ab78935c9e67002676652497c1e6c20c8e9b20de8b9501209f14ae |
