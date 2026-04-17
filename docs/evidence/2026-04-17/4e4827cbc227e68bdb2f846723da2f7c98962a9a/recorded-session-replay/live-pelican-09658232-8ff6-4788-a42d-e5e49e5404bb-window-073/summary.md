# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-073`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4344938604e067860a8ae5cde1fca1ccd4f50c2742543e1ed5dbbab203e23d74`
- fixture hash: `sha256-835a3394dac9a8b8023e71ca801b0ad86f7853cc9e826a2ddbfdbf3c56dd351e`
- score hash: `sha256-89a805f3e16e0ba61e543661709d25101ab71c059ba0e5e44ddf99ef1df0bb1d`
- bundle hash: `sha256-540d8625cc9bc66116277308a385ab5181257905bede1382900580eeda806a13`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-95f6b633b00bd779574dbd24baa772f0fb4eebc8350ac2c13ddc54230525a7fa |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4c22eaffe94c770f879e0bb776f0ec955fe013b709bc8b207d25ce2d2fa0f867 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-29bde339e4f4c1f44f6da990ab5c05294641d74d09239495dbcfd235424b5fee |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-57cc8dad544e33cc281c0bcf313f3b3b33af3e6ffddb10b8fc38e01a20626d8a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d13fedb2 | sha256-55902fe2183aa2c702caeb07c390a1bd22139531799725ed3dbf6f93473d22d4 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d13fedb2 | sha256-8f039e1c1a3e2626df223b7225794523ad1fe541b53772f48a5202c46da315d5 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-1a1be971 | sha256-792f406dd0a70ad27240de2002efb83565a1839ec395dc2866ca5673b6b40d74 |
