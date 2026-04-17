# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-062`
- winner mode: `graph_prior_only`
- trace hash: `sha256-68dd4afa8cf6968b41418bd460fed5641fe37e0c30a004be1adb6fd97d678410`
- fixture hash: `sha256-0984347c035679f491e5e5ce92160de0970752142af6bd7d0f80779707ccfa84`
- score hash: `sha256-eee4e6c17388ccc1da6da9d0ce0eeddff924551643201b8ccbe0d8846c03f7dc`
- bundle hash: `sha256-7742ec56c9b70a6251bbd5d8a8bd5fd3f243029818ca5a72746cf361bdbcd838`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4e7043d44034a818b042ec107f761c7c9e4d805591027e32242d8b764dc9d866 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d968832a80523c9d349f72ce1281e2569046d3cb59b7674ab13fe12332a3c31c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d24469a80d4293be19ad09c39ca60a86d499de3ca2359c2a83100d29b397b9b9 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-c1f239cef730ad3549dadae585016580892a2e31acabd08ea28157215033d7be |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-13a24131 | sha256-7611ddaf7da44490fd4bd2eac08fd299087d55968922ad9493fc8dad43694c45 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-13a24131 | sha256-888822aa63de867c2e9fe84491dbbdce2156224adc377cc2a5fef121edb44c68 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-21bbd88e | sha256-cc4823e80dde116fee46951c26ce2298fceca9c631877aad2b4438f9073009d8 |
