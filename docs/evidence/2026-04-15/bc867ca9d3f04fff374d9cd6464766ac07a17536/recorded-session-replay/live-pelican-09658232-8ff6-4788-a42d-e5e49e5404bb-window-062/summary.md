# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-062`
- winner mode: `graph_prior_only`
- trace hash: `sha256-68dd4afa8cf6968b41418bd460fed5641fe37e0c30a004be1adb6fd97d678410`
- fixture hash: `sha256-0984347c035679f491e5e5ce92160de0970752142af6bd7d0f80779707ccfa84`
- score hash: `sha256-ef39cb73ba316401f35ccb7342584b35e4d36ae0ecfc43187d94e02f2c780d64`
- bundle hash: `sha256-7d9ca1c47140ad5f17d5249caf70bcbd98de631bae67c22cf123cf89dc3e8d61`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4e7043d44034a818b042ec107f761c7c9e4d805591027e32242d8b764dc9d866 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ea337b1b6c7d56456d32fc5701da29d1d18381a8214c7d4da6535145fd73472f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-83fa3ef001cc66d0167e05662fb8e8456bbc2f05981836a4fd0da2b2baf95899 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-1a9127fce75ab2f808c100e0ac53042e58b9e6d26d539f8507b3240c3ea80539 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d7666b4f | sha256-207247ac858a88f4b15a01a573271e921c9db3e0ee04cb8ac317109e43e80eac |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d7666b4f | sha256-b399201cd59244e8bae8fe4e400afecd5bc567c9fb8e8ab13f2a2a586463a547 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-d7666b4f | sha256-207247ac858a88f4b15a01a573271e921c9db3e0ee04cb8ac317109e43e80eac |
