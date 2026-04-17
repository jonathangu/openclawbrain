# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-19cd6a701f3afe5404567d59955346d7cfc26c77deb7b29e61fccacc22d3bbfa`
- fixture hash: `sha256-4dda7357e5652f879faf39fc4f606d23e6674326c96ea6b533ba27ecfc72cf16`
- score hash: `sha256-ed943b644704d26435b0c04c1000f362d3b94653902d9292fb405d63d409418b`
- bundle hash: `sha256-60310f4a72df44b60fd7413ca4be4f9aba0b4b06f09db7fa1a60b64f6b806385`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-934729bde748377658ef5251e3c9784137a24d5cc133cff448c2ec475fa6a4b7 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-97e4165aecca927fd9fcbae6eeb676fbd610ad154b95bc05504fd6d36f16be9e |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c812c17be361cdc87257e49f92fddf1f8d0c5c462306f19cb6af7af916f4a1f7 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-5c2cbe39eaae98d36db2bcd72fe615a9390c6a72cf83539cbbef202a35b10e34 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-dfd64130 | sha256-f891001489f75f70eb1667ad34bd44492d8924484d2a75f09abf54a340f5dc15 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-dfd64130 | sha256-15ee7b3b76426cb55cbf55a1c1c1cf69821d672944a2d4caccbbe3b6353a317d |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-c3519e8f | sha256-336da4bd4f330002ce827a584beed27f0388f5a545bb1234c17c83f0977a0243 |
