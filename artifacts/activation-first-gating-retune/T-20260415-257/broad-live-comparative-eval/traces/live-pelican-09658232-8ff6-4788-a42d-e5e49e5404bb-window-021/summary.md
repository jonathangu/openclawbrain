# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ba20557b6de7502d32e3b83fc90ffd6f8ac19ac17c3a2b682f0895bf8bb69c7c`
- fixture hash: `sha256-024c24fcc3f69f4d62a086b795f3d8c9e3625b36454d26d1e235e1664a651060`
- score hash: `sha256-71cf316f0b29b79707daeae7822f55c6d674100c2c6b60d0898c57426fcff03c`
- bundle hash: `sha256-fccdb3ac2b5bd5505cc7289233939c0834bebea1087648f1ea91e42141a82d66`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-91ace2bc1ee23370cade1ee9720612db55ace83c09692395cb273562a40c2beb |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6514f1ced36e7a092e69c9a0951dc2b6cbccbe9851a2ed32ba405117643e79f1 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-5f5599fb5836fc5427069666336ed36bf05aaf2634386842ef46ec357b647b89 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-ad5eb1a7b2b2d980281e4bafee8cc6a6f3573ef5a288f8122691709b39e0e45d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-25af7da9 | sha256-eac5f5b8e8a49d4ad42399c0365b776cb0f80a98f8148c42c3eafd1fb6a1499b |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-25af7da9 | sha256-eee22c5332765d2d209f9f481e1ba1243f2cd21a8b7790299e0e2d9ac4884a82 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-940257d2 | sha256-4b34daa4ecab39593d5142efaba8b64f11fad4aa724ccdcaa3d71a7e6f97b0f6 |
