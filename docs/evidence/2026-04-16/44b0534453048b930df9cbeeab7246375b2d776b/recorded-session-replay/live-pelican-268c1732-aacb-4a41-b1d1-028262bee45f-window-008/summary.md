# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-821174c9e8199055ec82b211fb2d09a993a3651df37b6c8cbf4707f78e0465ca`
- fixture hash: `sha256-bcd47938b5aaf944e8ae47149b98261af7f5e6b78cefab4ab8c21ef4d0f8288c`
- score hash: `sha256-df615a4858e0f2578c8a2fad8ad7c1b4abf3cd8898f99b235986674cf8643d2e`
- bundle hash: `sha256-9eabd517a647d7514096e5b52af5cda91620a617f171648b6478b680b36c5a2e`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-46bef670c7b3d0d500dbccdbaeb44127bcccbda5425d78ea64b9256410c95a9e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3b8b418eb4f33d1c2a3828c17e7851fafcdd867656e14479f04227c0e322038a |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b82544965c221096ef1e09a5a645f28f3d0642793aaded750bf049c89987cfad |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-e71bad122ad294825350c48143c0225dfa55abac26158f0732eb829b40436ef6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c0fd36e5 | sha256-016add8e69e8e434809bed66c85bbc308f06b608920011cc648f2cccfad7244d |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c0fd36e5 | sha256-22521ae53cd356863e33fa22edc21476bbd5ee71bd2a9282456ddc3cab85b3ca |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-d8becf52 | sha256-28dbf93bf3f45a23f073fb1c43ea83861985fd621ca26b4ac1517ee403d9a227 |
