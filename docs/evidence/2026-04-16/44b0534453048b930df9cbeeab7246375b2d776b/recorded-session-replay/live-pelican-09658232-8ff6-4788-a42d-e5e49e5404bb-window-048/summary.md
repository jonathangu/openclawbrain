# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-048`
- winner mode: `learned_route`
- trace hash: `sha256-9c32a87b231e4d5848a772d9d1cb8d355e8b17c5c883fc0f1ca8776ef042ba2c`
- fixture hash: `sha256-66d4441e9cd89d5df06e129fcf70accf27e8123573950bf81a6f813e2979adc4`
- score hash: `sha256-aefac928a4f7e8aa5e5d6b926fd3e60cdabf4eed68e5d217a0028e25c30998e2`
- bundle hash: `sha256-349437f0cfd9664b98c8753fadce74ebeb129751811554dedaf8d25c73753e65`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 100 |
| 2 | vector_only | 100 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/4
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2b9d9843197c6ea9cf1bbaf94c65647f4ecfa1e2224f8678711a552cc896cd7e |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-3fb77e964da731781985378cfcb568db6ce7108edb7308db8bd9ced959a128e1 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-18374d977899cecc9b7bc67f37aa5d9af444a9d37ac8b100e1dc1ea348be7a69 |
| learned_route | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 2 | sha256-527c7d5878014b9a22e1fb2666e6594332e89154dbe3d6d72ab1784871dd1572 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-efdd4ead | sha256-61dc26520892de85728ae882e1bae404cdab06502604264cc149b56178dc58b3 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-efdd4ead | sha256-d76567205e869035b110e0f45a7fb80996818bd37e20da968c38dc373de90637 |
| learned_route | turn-1 | 100 | yes | 1/1 | yes | no | pack-3ef8374c | sha256-dec68763605ec1291de633a242bce7d7733d15b01f1a7ab8cd06d4ce5a38b01f |
