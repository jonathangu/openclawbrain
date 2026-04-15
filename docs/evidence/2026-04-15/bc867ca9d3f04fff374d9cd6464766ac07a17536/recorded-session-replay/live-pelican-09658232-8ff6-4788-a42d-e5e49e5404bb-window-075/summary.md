# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-075`
- winner mode: `graph_prior_only`
- trace hash: `sha256-31f15910a7f37f6942dfc1fa59eebabc12b733c2fdbc101bb92672de7f721f0c`
- fixture hash: `sha256-d3d3b7c9daea7f5dceb8bcbc7d0b182082662e4eea5368602c8cfc65a5234e7a`
- score hash: `sha256-157a3a86cc10d8920314a50097d01f09abb7ae2568005be9261675760f99293f`
- bundle hash: `sha256-1a1f4976604b2a23b90c66aac82af81a00933ed644ad468547f192974e7d386d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-eee7751ada66c814393120538ba88242a0ad04eb627a4b24f36524aa1be2a704 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0083b0c631c15097a54c7ba49561d06061b883deac556ca9557cfbca67594fae |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d214c4759372d0a423b9300c4be6f3ce73e0d110421bfe6801238779b4a244be |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-541b5ace733a14b74beac1c6813a6fb500f978251df9f6dded171e1a1fe6293d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cf85b5d6 | sha256-bcea20d9376272faa2b695987d6066f7e387637de5ece9155c6c943984adc672 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cf85b5d6 | sha256-0d025698bcbbc9f036c9ea920a2706f5a5dc0a4b079e0cd6ed2bb9216eecd108 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-cf85b5d6 | sha256-bcea20d9376272faa2b695987d6066f7e387637de5ece9155c6c943984adc672 |
