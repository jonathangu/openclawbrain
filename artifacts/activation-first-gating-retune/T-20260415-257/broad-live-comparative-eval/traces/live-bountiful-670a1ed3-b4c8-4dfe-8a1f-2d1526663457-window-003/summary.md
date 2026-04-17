# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c77161fd107da0850cb368a9be7a432917f75dca7a822871991fd4fdb28ea1b9`
- fixture hash: `sha256-305a6c8327f5df890119bbc3711133fb545cdd219a8e22832dd1a9b40c670ed7`
- score hash: `sha256-a4b2dc6967f384d56d75ff75dbe0f555360292e4d1933a2a20e828fa84eb1908`
- bundle hash: `sha256-45e9ce2ada129d47b8d930da33a364461eec85d55014c2c78d149793fe0e70cc`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f9994dc36549b50fc869a0d07853fce421c050985a49a6ae0b76d1bc12cb356c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f5c52af5ac962ad053e8320f57436cbec89302d095d29d06b934544aca134343 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-95fd82dca9c3b8b0ec821030ac30a5ee2b608b993ff4c1cc636b6cb17c61d7af |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-daf37a5ac92e2d0e00ac6c5271bcab23c08c432122ad0dd888cd7a2cc32e2d34 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0fd77078 | sha256-866f80933a6a1f024faccb69d9a5b1a45fbb6bc1d578e5f4252e3b8cedd6e20c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0fd77078 | sha256-6290bbf83c2fb0491f4c66bb4d9986cd83adc037e06c80b235e3502086c86ae0 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-5d3db3bf | sha256-74e873f8d84e52ac965cf626d094b87719ef308c544267202c79ef0e48082bfd |
