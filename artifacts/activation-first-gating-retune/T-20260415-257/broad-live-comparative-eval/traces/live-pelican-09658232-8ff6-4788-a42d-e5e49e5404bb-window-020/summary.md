# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-020`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7f3b511c142861747542dff1ddae4669701bc9656bef363a96e4508cee5f2a20`
- fixture hash: `sha256-2db80cfb229c04864b42f8f3b0cbec60d6dc032d77659032291a70b2cac64512`
- score hash: `sha256-547b9ce8ef6eac48c830ffd2710e54b47374fb26390b537736f43bb2af013590`
- bundle hash: `sha256-bde5464a660da428fff050782dec57e48d043080475a9a282c5597b051ad3349`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-03f1fe0b1ca7c86742bc307098d07423af66afa6b8715bd5d40ceee92e59b30f |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2c3c732c90f019a86a99beea4ce5e0da3dd4af5d59042758b54ecdad225c9520 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6b94d9880b82b53610793275eef838b352e88f963f4a6c1fecb0eec07513883a |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-8928afea24e99b94e0c910e7e702065d5e35bcacf272850883a402f92c22cf79 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-770559a6 | sha256-45e29ccac0fab831cb5b00d4f74ddf13b87d6fe13ffa327812c0ed65f763beed |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-770559a6 | sha256-02e7fd97263214ccc521303f3f7c3842ea00c20e504da6d120f2bdaa20999978 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-a37567bb | sha256-9e30c362618ec1465d4cdd414cac55db6c6d66d835b0e08a4321e22a73a7558b |
