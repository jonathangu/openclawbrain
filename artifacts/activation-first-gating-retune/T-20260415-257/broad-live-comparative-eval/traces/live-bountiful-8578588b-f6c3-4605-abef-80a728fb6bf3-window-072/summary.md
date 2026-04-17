# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-072`
- winner mode: `graph_prior_only`
- trace hash: `sha256-816a27f51a89ee3ab61ecb9dcf7fe22803e339036dac94e7aa31864fd0968283`
- fixture hash: `sha256-c096ed79c46bcc788a54db7b73d6166a0d80aa4dd8f8479c075722de69b2b170`
- score hash: `sha256-25da4ce903b520416dbaaa3311b38de00ea104e9e5c3679e5a4c2ca476a6e2b8`
- bundle hash: `sha256-f032b5d4cfa519a0fbd61eadd8cf7cafe4b9502021a6f6a6a1a781dec579a3fd`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b145c71aa4e121e88e077c9e9aba7ef5e72b3964bc6945ed620719f5b1c0299d |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-491e4d7dd626ac97042f73bbdd61514f72068ab02f4aa64e40f5672b2724fd62 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fab223f34e73ed585cb2d6a2c44d98489921f2edfdef4e4f6c4797df1f98bc97 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-861fd57552084043c34fb68be2b072dd20e622f1b24282bc6b88665e343365eb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-7b256851 | sha256-abf4662ad404b9e84c542797db4372ab8a5c0188ef17b5750752c026a0d93df6 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-7b256851 | sha256-ba01b65c483537864bfc16d7b53115af9f0543dc94a9075c2e2d97f22735c9f8 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-97a8457a | sha256-4ec05e4f229854cd5d184ca38fa7f1d664a420384414056c047392f952f39595 |
