# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f6225db554a4364f98319fe1020858f0210ac404609a5303b38b0c9b7d31d658`
- fixture hash: `sha256-ed619df46c192591981773c10de0d36b74c9e5dca7f78e3181e7b1aa8b066c66`
- score hash: `sha256-8f83f993a6ced3c180c0a01ac505d409abc20a44bd862ed42ae93d54cf739dae`
- bundle hash: `sha256-3a012d0c910919ca06315fe647d8c0470e31ce146165dbaf9f31491a8efeb330`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-88c859d8e45ae0f9acc890628cb16be0104a115fdd6d7748c85da1fbfab29bae |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9aa3d983b7f6c6ad1eed8fe38cfa0fe0cc0bc561ef35ec6e03a22157c6185065 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-be27702715f8371d329ff45869db8187d0baa99460c95f9a3e1b677797294476 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-c5bb017810ce9f3f2b10f15068083a8310fc9c6d2086d4daf9cdec5aebec6dd3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ac0b286e | sha256-3ace61774c1469eac11abbecd5723e79445d4b33b1933815b78a319eff25d2a8 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ac0b286e | sha256-3ace61774c1469eac11abbecd5723e79445d4b33b1933815b78a319eff25d2a8 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-ac560a55 | sha256-34369ad07be249e3d118eba1302263c0ca71f87a392d81433f20558bc261b0c4 |
