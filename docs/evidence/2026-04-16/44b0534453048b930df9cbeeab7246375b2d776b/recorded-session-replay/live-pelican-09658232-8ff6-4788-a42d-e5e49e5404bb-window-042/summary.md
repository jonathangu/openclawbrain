# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-042`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8470650c25739a12e09f620484c19ed76e535bf12ac60fa5bc19c4d9e71da263`
- fixture hash: `sha256-1a290f6c39ed84b2ca073e21a57823e82667ab7d1408676870645010e286d76d`
- score hash: `sha256-330ee603b6016a6d6b4db956cebd0a923e9cca6d1354305831482a5dc38ed9f3`
- bundle hash: `sha256-2463504f2a2d2cf697849a0e30219e85824533125b87a6e6bcb2700833581fac`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-575c5da35d27014ccbfd8fb043d25f84e1287b134d1db92502cb2f005c370afe |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-48610ce9974c7cc313ceea50b9bc1b62ef0b0b336f9ea80651875a8d9655b727 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b39b940529a87c2c66faf6355d90df3350e5f42ec50b7514088ab3651b33874c |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-6cadd4f6d32f3332c643839d835fb8a677b81008909d19cd7baaada29074f824 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ba02f526 | sha256-61a9fe7b41eab8451aaea4b15b3d286cf53945ae97e20e32e7e355e9ec39813b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ba02f526 | sha256-0bb44a5f84b47f83c35e4aa9d5e940c813f967f3f1e087321dfc4cc792ec9175 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-8490e373 | sha256-4c04ca92ef177873f380679aac93bc5914527ea1735e2b25762f0133f0c6135c |
