# Further info for modelers

## Prerequisites

* The model calculations require [draf](https://github.com/DrafProject/draf) v.0.3.0 and [its Conda environment](https://github.com/DrafProject/draf#quick-start).
* further dependencies:
  * `dotmap`

## General info

* Default units:
  * Modeling time frame: 5 connected reference years
  * Modeling time step: 1 hour
  * Costs: k€
  * Energy: kWh
  * Power: kW

## Model formulation

### Indices / Sets

* $t \in \mathcal{T}$ - time steps
* $y \in \mathcal{Y}$ - reference years
* $r \in \mathcal{R}$ - H₂Ds
* $a \in \mathcal{A}$ - H₂D-consumer
* $h \in \mathcal{H}$ - heat temperature levels
* $n \in \mathcal{N}$ - cooling temperature levels

### Example formula (... if this is wanted)

$$
\begin{align}
  & \dot{H}^\mathrm{Dem}_{t,r} = \dot{H}^\mathrm{Tra,in}_{t,r} s^\mathrm{Tra}_{a} \eta^\mathrm{Tra}_r \quad \forall t \in \mathcal{T}, r \in \mathcal{R}, a \in \mathcal{A}
\end{align}
$$

