# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c178ada-4f98-44da-9ab2-6ca13f2e2441-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5c1146574706ec395e6f5011fe3bdf3e510b31ef69670a55eff27bc156061d1f`
- fixture hash: `sha256-b0b87869202da9099b109d7a7b86f16484e8b3960b663b22dcb9b0c0fd925784`
- score hash: `sha256-4a4fa6c26b47d58c027614f0ce43b9aa8c31d723265a65891480293fc0712f17`
- bundle hash: `sha256-0ba09a4feb1c692fc808273c8f54b225c22ecad2757883e57e2816d447b7ebc9`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ec4d1011b47b3f4bd2e4eb4b5459f997ca8cea571b8e6f78dcd457b660861c55 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7789b719f05d5f5ce05d869d585e82a98fb3fb3f19cd07c75cba2b9389658448 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-a1e20da107f163399005c17e86361023c3e2a14624bfc53be2c615b0770ad097 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-25931013 | sha256-58fb9c3ed47cafe7bf139bc6a967d6fbd6786349f274a6f350536a6e9ed8854c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-25931013 | sha256-1bc995666e4ccdc6a351472f2714bcfbf4433cacf9321750070f2c1e05798ac4 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-a540f586 | sha256-ec70d9b7e2b3e062b802a401b627c37272abd4fcbd031bc5ee22e700b896fd57 |
