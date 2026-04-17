# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-036`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d855fd526b0432f6da4ae83a914585ce6467161a22fe45c628b20919e2994b08`
- fixture hash: `sha256-a059e9b8611b556f3c483b97168ab252147668d3316414532e38d0791f5cd0c4`
- score hash: `sha256-3316c05f3f977f17032adaf10915519cad61261b9e416c01ccc466b018b8eca0`
- bundle hash: `sha256-774a3109ac9ebc3f85e2c41e7406fcb500bc406cbb94dfecb16164dff2c78af8`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-46808c7f90eba103441fec044b9224d9dea48b85cde7d0c53efec734a800db3f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-78c3b9200b2b0cf9f9d327d15eeef8d0d4e1bf136d5f6a23d6159e5a970ac94c |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f382bd2cce51151b24c1eeb8eb4d79132f1d6712474f01ffd73c54c6ad489d43 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-47bf7ba9ff6c52b6a2c883fa11e60f9091468bb11e4b7d56fd9440d781c74581 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-faa5e15c | sha256-6d335db6f2e895a7098661d333986df92fbb36617e23742c358a4f4389d18825 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-faa5e15c | sha256-20c3fc47ba50717b1de4d0ae1782e1f1237eef75cd07489086fb74d2b90c5020 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-5887252d | sha256-e1d151af01161f6c14fb98f6d82e33b711edd5a10fbd97803f594d4787d40a1d |
