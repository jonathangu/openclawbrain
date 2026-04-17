# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c178ada-4f98-44da-9ab2-6ca13f2e2441-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5c1146574706ec395e6f5011fe3bdf3e510b31ef69670a55eff27bc156061d1f`
- fixture hash: `sha256-b0b87869202da9099b109d7a7b86f16484e8b3960b663b22dcb9b0c0fd925784`
- score hash: `sha256-d0d8e9085addea38057bc5f702f0b083a945e39a9e403ee9934d9e330d3d4ec2`
- bundle hash: `sha256-a61ce97093ca9a42f7e9c3a754b07b3dd02e344b7bc3808ababad61c560b0569`

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
| learned_route | 1 | 1 | 0 | 1 | 1 |

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-aa95a9ace80ed876a0f8c1c80f7aa67e61fb6997d4b29b558d664aa27fdabc10 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-93daa1593bf64aa20abd7844ad1d54a5044e30e1acdbee91d2b9db47f43d5349 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-857d907b82cf78064d6a81b84f610dcb96935d88efbe67ca6c40e54d0d2bebf2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ef2aab9f | sha256-c3bd3c6df77a3ccb9d5df9627f31621e9a1ea3feb58867a768885b9307d30d1c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ef2aab9f | sha256-c343de9ac961774de09e333e63af8093541fee84d271a98ac4d085159aefaa7b |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-6ed89112 | sha256-58eca23e88d5cb01d9dc78592ef0bfdb248e9d8f0b95ec50c904a5c341d05180 |
