# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-020`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7f3b511c142861747542dff1ddae4669701bc9656bef363a96e4508cee5f2a20`
- fixture hash: `sha256-2db80cfb229c04864b42f8f3b0cbec60d6dc032d77659032291a70b2cac64512`
- score hash: `sha256-9e9682fa139727ff0a56c48cb2f873de3cbc7cc7a4933be1d9568c8f81284dad`
- bundle hash: `sha256-bf8039bcb9a46709832eb6c8f27d7bec9b1042e67d52687eabc01672c55654cd`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fb0b755d9e4fce4547a39a46e28d40e16ffc36eac7774a8255ae7c0382b6c1a6 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6e585d44ad0a00ee7e5be0446faad07c9c88d399b96d27f6d25a4fc057627c5c |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-dd35f9d56dce4935076ae9b5413ba4dc9bc4ac675cfa263ca8b39bf075b73adc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-25552715 | sha256-5b5f304d4d8112ee27dc473c2f0cdd9146a92f68d1473bcc0fe7bebbe3fec625 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-25552715 | sha256-202a89849f9a3f6ba6a1cb894070ab16bcfdb547915b822b7034a60249ad5625 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-51c5352a | sha256-40650a1be678b3aec8b8b1e5264fc2519b44cdd1ef503ce70b46cf89dcdf988c |
