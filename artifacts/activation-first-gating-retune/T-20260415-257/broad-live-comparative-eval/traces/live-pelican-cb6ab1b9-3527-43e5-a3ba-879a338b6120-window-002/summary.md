# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4ecb7ad01ebf51dd3aeb7754e784e70eea9f4067a9392ad81778aff88de83b03`
- fixture hash: `sha256-ad7aadbe694390cc07af980435b05bd2086d5294c79bda5f4f75ff348a4a3b75`
- score hash: `sha256-8aa57d5958300d24ebcadfa42fd43065e3aae7e82dc6e40ead339276fdfbd661`
- bundle hash: `sha256-f69cf08c082c11bf8778ab5324c5dc1ed018d193b61d7bf8206291826320fcf8`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b8a8a0e6d7dd1a7545143681fd0202299acfcab2ad5ce85ed5e5cddd516c7f67 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0a283b3a691fd587fcbab33c7b2d56440c2c3a3feae549615dc036aab1f3d042 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3c345339021a87f134d3e840e29aea8ea9d00fc5686c6bc37b4a447906f4a23e |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-81b56d3d099ca065616a314ab844921cff7326f1068eb47178677a1779823ec3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-34b777ca | sha256-8b841e92a4992c9c0fcdfd5bbfb8e56b892020cfa58ff75f574b909a1e6670fc |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-34b777ca | sha256-dc756509fc20d703b24e5b260615cb2b02ac31213570621d61d4623b67259db9 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-d4a996fd | sha256-a319384d091bd9ec161fbbf9dd8dfcdf48e31390cb918319dff575b2bf247829 |
