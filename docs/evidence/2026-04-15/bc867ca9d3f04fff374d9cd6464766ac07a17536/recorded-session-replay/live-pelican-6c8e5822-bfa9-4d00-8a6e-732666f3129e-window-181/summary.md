# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-181`
- winner mode: `graph_prior_only`
- trace hash: `sha256-50cbfb4de0d67c0910ccd1f15acc9132454b767d6a9ef6092fa51c701d086751`
- fixture hash: `sha256-ed982aae33c06dfcffb629c09975a63d396b69570ab9ad349366a4a66aa757f2`
- score hash: `sha256-ef7387082139ab76c8a5587bc6860757256a6a75679046d69a19e9358cc3915a`
- bundle hash: `sha256-2dc4763969b55338418968bd0dda013ddf4a4cbd116a7473a25005a52c37d7f0`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 0 | 1 |
| learned_route | 1 | 1 | 0.666667 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9ce8acab11edc00f581b930ddd46ccaeed311548b8f75f0398d0e21fa5078567 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-887d32f86a8648b3ed0978b807db9ad7a771fd326495f1b70e652e61c8d52e47 |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-9a3107f6e065a6d66c0e456dc4e8b9acbd8564ad049f966201465dcae9fa2a55 |
| learned_route | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 2 | sha256-5c51f07208a4b77666c6b1c1677ede3728561d13d5b5353cc38d4a60d288f0e6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-74262922 | sha256-518d83f125a89135459bde152c3138c72ba03e22c77cb35316e389159a9b11c3 |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-74262922 | sha256-0778381943e6594c1871cbfa1c9741c58be045b90093406ef63f26e1a9fd374e |
| learned_route | turn-1 | 80 | yes | 2/3 | no | no | pack-74262922 | sha256-518d83f125a89135459bde152c3138c72ba03e22c77cb35316e389159a9b11c3 |
