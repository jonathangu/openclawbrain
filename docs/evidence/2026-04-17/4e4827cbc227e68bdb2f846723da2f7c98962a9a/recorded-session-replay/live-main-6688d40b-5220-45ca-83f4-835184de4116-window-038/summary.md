# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-038`
- winner mode: `graph_prior_only`
- trace hash: `sha256-97f203f8cf6e54b5353937fa7c2a9de19dc80e0c9cd1f7a7efe923af5d952db1`
- fixture hash: `sha256-125dd958dc76d20d11fb8d8f175ae0fed91a68b28d91d5171d2943217403837a`
- score hash: `sha256-e6d88a4356e14e1c45426dab6ef5c91551e88eb27de8b4579706ba212e017856`
- bundle hash: `sha256-bbed7d8e736ffb9b36fb35795075d54127f1be0d8966cea7c5e2daf8554ec832`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-511d0fcc64563746c6c18b192b94f28492b7e276c306dddc6df1e62d381fef89 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-94f6d6e04ce03492046385b79f7f75a4f2772f0e5eaae9083dbd09d0341c3a84 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a0a383675cf98441e23b15fc2a97374948bcc93267f52dc808d05c92b105d1fd |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-4dc0a97a3a6f4cacb9a66d78cf0f2cbcf76c471b0764515d80f5e26e21089d06 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4fa4f7eb | sha256-c35eb4e61c0d40d27531c9888923b52fdd7c4eab1e8f758c8f42a0f21c675f71 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4fa4f7eb | sha256-650b51339b0e15722ca4e7016026092bc34e5779656bf63e5101a295b970afe6 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-5dde96cc | sha256-bc7e9281af8b7a410929690e3a6a5ca22ed8f1d8f9babcc01bd4eabdf4a32141 |
