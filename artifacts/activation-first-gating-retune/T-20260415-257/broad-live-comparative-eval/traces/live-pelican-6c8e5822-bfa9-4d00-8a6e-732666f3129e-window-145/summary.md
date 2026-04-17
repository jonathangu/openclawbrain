# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-145`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8588f62e6cb39b6bbebdb00e938513a4cbaa506b41be87532a11b4304976dd66`
- fixture hash: `sha256-523f979d52f465f7796de01a235f1b7bbea1b624b0a2f4aa71ab4b02e1ae0958`
- score hash: `sha256-8a28a2a51a25f92d35e50c0f3341a2aec41935a6e8b5e33338547ac68840bbc3`
- bundle hash: `sha256-b83cba5e6d8bed449c730618066505a783ae18dd9f3c67df19556075921cd05d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-172463d69f5ae184c08f379b77a680b592819857917ad8f3596af66f22037f0d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9b230934667fe394926512ccd6da536a97237b37f01b868f70c44fee0f3d9729 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-87907092a80779c34b3a9b11c8f0bdd33175534445aaf5c1a96613902e15b877 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-99923234e576995e5ff85568a0721128623aa4b4072220cfb53e675cb34df9df |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-33d41434 | sha256-fbde8a756b844202d86c797fc4c29ae11fec2895f7a3318b31841426b6e60866 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-33d41434 | sha256-4b7279be1e49ff8cf85edfa9a825890054581cafd667fc7b27673bf3149b0991 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-5408f0c3 | sha256-ba3cf742c1f755a174170ec4937912a2136cdab9c11f39fdd90bbfb9a8d617e1 |
