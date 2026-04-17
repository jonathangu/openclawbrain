# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-041`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cd17705850f5fd87f770e4757922f483be90c3dcc5bfff44d696c49e62560cb7`
- fixture hash: `sha256-743937076adce554085fa9dd3236567f573df76180477a11d06a07f43c4044bc`
- score hash: `sha256-1e37df0fe8d2b6fa147d99e5092d4279507fbdff8f54c2bdef88a5f7233fbbe3`
- bundle hash: `sha256-94c0da2073c0fb965e9996f312a26e955e54ab31da7515b486a7082e9f3c4ff6`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-38ffbd4329a21a765f40f1a44ad7d1cc0603504c91e4e697e7b573151d0b2478 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-41c789e0a3dc9b337148ded2f1268f9fd3d0af429079a4aae42322e4593f6a83 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-21ec7946400ecd594bad63acdb03e72a4ad7df57b6d23e0bfc1ba6912a990898 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-646bc8f9ba4a9d6c2ef9723084b482722605c430c5138d7c68f75a4010d3070a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c77b3e68 | sha256-265b911651bbf1b0d90409a1c8d4e99e7349c97198556a1ce97048f16b897f26 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c77b3e68 | sha256-5872243b99941a7f33fea2e7299e8ceecc5d84137a54fdfc3733520ef1d0f71f |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-072fcb41 | sha256-5b2d1260eca434193ce6df406d0c5eca27acff7d286d2c62d83f5ef66a3683d2 |
