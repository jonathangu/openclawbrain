# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-084`
- winner mode: `graph_prior_only`
- trace hash: `sha256-57d54d463c3d335756b9ef845ab48b21d6ba79bd455096740f5eec6ab5dcf52e`
- fixture hash: `sha256-c975a7548913ddc09f78bdcf8d6f035b2cb79bee5a8fff204c28b6e92be5b531`
- score hash: `sha256-3b0f66c76eb5a3c9e0eec2529500dc791ee4ad6021eea795c3157c0e4cba802f`
- bundle hash: `sha256-c09c29564f437a77a6fb8ea924d809239d558cc4f4ea8e781d8739dbe552c7b1`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6f565862470853eb1b48835f5dc58d5e78705c4b54f6971c4806d12966cc7447 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-44a8df767a1a673b54c01735b42d0ac55bbf6c265cefaa15596cc4fbb4fe0b48 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-aa079a7a78ef716ad873a910ca6c85c6e7d057d0742cddb046b2951eed002075 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-41656b2fe80c32284f98633796b5747c9dc129fe2fbabc8a60b88258a087ab3a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-97e22c7d | sha256-3cdd42914b6b89916b5802b569f58a2317cdb39ddd6335362424b24059776347 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-97e22c7d | sha256-9b3510f7684075ab46c9c999726e5e0dabc62f3dd3e5b1637f54ad932f7288c2 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-97e22c7d | sha256-3cdd42914b6b89916b5802b569f58a2317cdb39ddd6335362424b24059776347 |
