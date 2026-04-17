# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-80f4bd70b8f229336838d17a92921bfca64745f162a9177679361b11e355a256`
- fixture hash: `sha256-1a25c630a19d83ff4b3784d9a97b879e228a965826872ffe6bcc2e6453fbac5a`
- score hash: `sha256-dec7294f17935e31c7517e5538b49490cb4a5cb086e844e1add78d28bd56b9f1`
- bundle hash: `sha256-de6cf2a6782919e29563f3da904b8400ad5a0794c9aa953b4015318b6bb830be`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-94ade431a5c986254405c71afb4d4071b897c04f7cbfe57133fcaa9500ad06d1 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c1770a5c75ca40bd63aeeae5e69053bdae5c695b58716acb8ce4f6b923b838f3 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b14dbef432420558b91fcbef9f25a7bd009e1cccfec0b28a11054007bd61ba1c |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-0bf56be343e269990e3cb7edaf628019ef99a1f08b9f28d3b934d18037bc6c98 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d0006997 | sha256-49b9cec4fefa14f773f2ac529c640b2a9b8c6f5c4b0231a3b906f6bdf41475aa |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d0006997 | sha256-0e30bbabcd9dacf34d55abc40a6844dab231f8457c3702e8b65f047bc2ebad35 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-ffda23ea | sha256-d8fbd57a24d09703ea763e6bfcaa802a09a17d2c83c7c8a87247e88b3caa3941 |
