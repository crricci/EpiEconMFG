# EpiEconMFG

Codice Julia per un Mean Field Game epidemiologico-economico con agenti
eterogenei per ricchezza/capitale e stato epidemiologico.

Il progetto contiene due livelli di soluzione:

1. Il problema dinamico completo, in cui sia la HJB sia la Fokker-Planck sono
   dipendenti dal tempo. Questo è il problema principale.
2. Un problema quasi-stazionario, in cui la Fokker-Planck evolve nel tempo ma a
   ogni data si risolve una HJB stazionaria dato lo stato corrente della
   distribuzione. Questo non è la soluzione del problema dinamico completo, ma è
   utile come benchmark, diagnostica e inizializzazione del solver dinamico.

La versione corrente include anche un costo monetario esogeno del vaccino
`ξ(t,k)`. Al momento la funzione è costante ed è controllata da `p.ξ`, con
valore di riferimento `0.001`.

## Come Eseguire

Da shell:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); include("main.jl"); result = run_dynamic()'
```

Da REPL Julia:

```julia
include("main.jl")

p = EpiEconMFG.MFGEpiEcon()
F0 = EpiEconMFG.create_test_distribution(p)

result = run_dynamic(p = p, F0 = F0, show_progress = true)

EpiEconMFG.save_all_figures(
    result,
    p;
    outdir = "outputs/dynamic_figures",
    with_surfaces = true,
)
```

Per eseguire il solver quasi-stazionario:

```julia
include("main.jl")

p = EpiEconMFG.MFGEpiEcon()
F0 = EpiEconMFG.create_test_distribution(p)

result_qs = run(p = p, F0 = F0, show_progress = true)
```

## Struttura Del Progetto

```text
main.jl
    Entry point interattivi:
    - run() per il solver quasi-stazionario
    - run_dynamic() per il solver dinamico completo

src/EpiEconMFG.jl
    Modulo principale e lista degli include/export.

src/core/parameters.jl
    Parametri del modello, griglie, prezzi, distribuzione iniziale,
    funzione esogena ξ(t,k).

src/core/diff.jl
    Derivate finite sicure su griglia del capitale.

src/core/aggregates.jl
    Lavoro ottimo statico, aggregati K, L, LI, prezzi w e r.

src/solvers/hjb_time_dependent.jl
    HJB dinamica backward in time.

src/solvers/coupled_forward_backward.jl
    Punto fisso forward-backward del problema dinamico completo.

src/solvers/hjb_stationary.jl
    HJB stazionaria usata dal metodo quasi-stazionario.

src/solvers/fp_kfe.jl
    Generatore FP/KFE, politiche, forward equation e implicit Euler.

src/solvers/coupled_quasistatic.jl
    Loop FP forward + HJB stazionaria ricomputata lungo il tempo.

src/visualization/plots.jl
    Funzioni per generare le figure.

scripts/
    Script di debug, smoke test e diagnostica numerica.

docs/
    Appunti di sviluppo.

outputs/
    Output numerici e figure generate.
```

## Stato, Controlli E Oggetti Numerici

Gli stati epidemiologici sono:

```text
S = susceptible
I = infected
C = contained
R = recovered
```

Lo stato individuale continuo è il capitale `k ∈ [0, MaxK]`.

Le densità sono:

```math
\phi_e(t,k), \qquad e \in \{S,I,C,R\}.
```

Nel codice una distribuzione a una data è un `NamedTuple`:

```julia
Ft = (ϕSt = ..., ϕIt = ..., ϕCt = ..., ϕRt = ...)
```

Le value functions sono:

```julia
V = (VS = ..., VI = ..., VC = ..., VR = ...)
```

I controlli individuali sono:

```math
c(t,k,e) \ge 0, \qquad l(t,k,e) \in [0,1],
```

e, solo per i suscettibili,

```math
q(t,k) \ge 0.
```

Nel codice `q` è anche troncato superiormente da `p.qMax` per stabilità numerica.

La funzione di utilità corrente è:

```math
u(c,l) = \theta \log(c) + (1-\theta)\log(1-l).
```

Per i contenuti si usa `l_C=0`, quindi il flow payoff di `C` non contiene il
termine di leisure nel modo standard degli altri stati produttivi.

## Aggregati E Prezzi

La massa di lavoro infetto che entra nell'esternalità di contagio è:

```math
L_I(t) = \int l_I(t,k)\phi_I(t,k)\,dk.
```

Il capitale aggregato è:

```math
K(t) =
\sum_{e\in\{S,I,C,R\}}
\int k \phi_e(t,k)\,dk.
```

Il lavoro aggregato efficace è:

```math
L(t) =
\eta_S \int l_S(t,k)\phi_S(t,k)\,dk
+ \eta_I \int l_I(t,k)\phi_I(t,k)\,dk
+ \eta_C \int l_C(t,k)\phi_C(t,k)\,dk
+ \eta_R \int l_R(t,k)\phi_R(t,k)\,dk.
```

Con produzione Cobb-Douglas:

```math
Y(t) = A K(t)^\alpha L(t)^{1-\alpha}.
```

I prezzi competitivi sono:

```math
r(t) =
\alpha A K(t)^{\alpha-1} L(t)^{1-\alpha},
```

```math
w(t) =
(1-\alpha)A K(t)^\alpha L(t)^{-\alpha}.
```

## Problema Dinamico Completo

Il problema dinamico completo è un sistema forward-backward:

- la HJB è risolta backward in time;
- la Fokker-Planck è risolta forward in time;
- prezzi e aggregati dipendono dalla distribuzione e dai controlli;
- contagio e vaccino collegano le equazioni dei diversi stati epidemiologici.

### Funzionale Dell'Household

Per un agente, il problema continuo corrispondente è:

```math
\max_{c,l,q}
\mathbb{E}
\left[
\int_0^T e^{-\rho t}
\left(
u(c_t,l_t)
- d_I \mathbf{1}_{\{e_t=I\}}
- d_C \mathbf{1}_{\{e_t=C\}}
- \frac{\gamma}{2}q_t^2 \mathbf{1}_{\{e_t=S\}}
\right)dt
+ e^{-\rho T}V_T(e_T,k_T)
\right].
```

Il costo monetario del vaccino non entra come disutilità additiva. Entra nel
vincolo di bilancio del suscettibile:

```math
\dot{k}_S =
(r(t)-\delta)k
+ w(t)\eta_S l_S
- c_S
- \xi(t,k)q.
```

Gli altri drift di capitale sono:

```math
\dot{k}_I =
(r(t)-\delta)k
+ w(t)\eta_I l_I
- c_I,
```

```math
\dot{k}_C =
(r(t)-\delta)k
- c_C,
```

```math
\dot{k}_R =
(r(t)-\delta)k
+ w(t)\eta_R l_R
- c_R.
```

Nel codice si indica:

```math
b_e(t,k) = \dot{k}_e(t,k).
```

### HJB Dinamica

La forma continua della HJB dinamica è:

```math
\rho V_e(t,k)
=
\partial_t V_e(t,k)
+ \max_{\text{controlli}}
\left\{
\text{flow payoff}
+ b_e(t,k)\partial_k V_e(t,k)
+ \text{transizioni epidemiologiche}
\right\}.
```

Per i suscettibili:

```math
\rho V_S(t,k)
=
\partial_t V_S(t,k)
+ \max_{c\ge 0,\;l\in[0,1],\;q\ge 0}
\Big\{
u(c,l)
- \frac{\gamma}{2}q^2
+ \partial_k V_S(t,k)
\big[
(r(t)-\delta)k + w(t)\eta_S l - c - \xi(t,k)q
\big]
```

```math
\qquad
+ q\big[V_R(t,k)-V_S(t,k)\big]
+ \beta l L_I(t)\big[V_I(t,k)-V_S(t,k)\big]
\Big\}.
```

Per gli infetti:

```math
\rho V_I(t,k)
=
\partial_t V_I(t,k)
+ \max_{c\ge 0,\;l\in[0,1]}
\Big\{
u(c,l) - d_I
+ \partial_k V_I(t,k)
\big[(r(t)-\delta)k+w(t)\eta_I l-c\big]
```

```math
\qquad
+ \mu\big[V_S(t,k)-V_I(t,k)\big]
+ \sigma_1\big[V_C(t,k)-V_I(t,k)\big]
+ \sigma_3\big[V_R(t,k)-V_I(t,k)\big]
\Big\}.
```

Per i contenuti:

```math
\rho V_C(t,k)
=
\partial_t V_C(t,k)
+ \max_{c\ge 0}
\Big\{
\theta\log(c) - d_C
+ \partial_k V_C(t,k)\big[(r(t)-\delta)k-c\big]
```

```math
\qquad
+ (\alpha_{Epi}+\mu)\big[V_S(t,k)-V_C(t,k)\big]
+ \sigma_2\big[V_R(t,k)-V_C(t,k)\big]
\Big\}.
```

Per i recovered:

```math
\rho V_R(t,k)
=
\partial_t V_R(t,k)
+ \max_{c\ge 0,\;l\in[0,1]}
\Big\{
u(c,l)
+ \partial_k V_R(t,k)
\big[(r(t)-\delta)k+w(t)\eta_R l-c\big]
```

```math
\qquad
+ (\lambda+\mu)\big[V_S(t,k)-V_R(t,k)\big]
\Big\}.
```

La condizione terminale implementata oggi è:

```math
V(T,\cdot) = V_T(\cdot),
```

dove `V_T` è preso dalla soluzione quasi-stazionaria all'ultima data quando
`dynamicTerminal = :fixed_quasistatic`.

### Risposta Del Vaccino

Dal termine in `q` nella HJB dei suscettibili:

```math
-\frac{\gamma}{2}q^2
+ q(V_R-V_S)
- \xi(t,k)q\partial_k V_S,
```

la FOC interna dà:

```math
q^*(t,k)
=
\frac{V_R(t,k)-V_S(t,k)-\xi(t,k)\partial_k V_S(t,k)}{\gamma}.
```

Nel codice:

```math
q^*(t,k)
=
\min\left\{
q_{\max},
\max\left\{0,
\frac{V_R(t,k)-V_S(t,k)-\xi(t,k)\partial_k V_S(t,k)}{\gamma}
\right\}
\right\}.
```

Quindi il termine con `ξ` è anch'esso diviso per `γ`.

### Fokker-Planck Dinamica

Definiamo:

```math
\nu(t,k) = \beta l_S(t,k)L_I(t).
```

La FP/KFE completa risolta in avanti è:

```math
\partial_t \phi_S
=
-\partial_k\big(\phi_S b_S\big)
- \big(\nu+q\big)\phi_S
+ \mu\phi_I
+ (\alpha_{Epi}+\mu)\phi_C
+ (\lambda+\mu)\phi_R.
```

```math
\partial_t \phi_I
=
-\partial_k\big(\phi_I b_I\big)
+ \nu\phi_S
- (\sigma_1+\sigma_3+\mu)\phi_I.
```

```math
\partial_t \phi_C
=
-\partial_k\big(\phi_C b_C\big)
+ \sigma_1\phi_I
- (\alpha_{Epi}+\sigma_2+\mu)\phi_C.
```

```math
\partial_t \phi_R
=
-\partial_k\big(\phi_R b_R\big)
+ q\phi_S
+ \sigma_3\phi_I
+ \sigma_2\phi_C
- (\lambda+\mu)\phi_R.
```

Per `S`, il drift include esplicitamente la spesa monetaria per vaccinarsi:

```math
b_S(t,k)
=
(r(t)-\delta)k
+ w(t)\eta_S l_S(t,k)
- c_S(t,k)
- \xi(t,k)q(t,k).
```

### Schema Di Punto Fisso Dinamico

Il solver dinamico principale è `solveModelDynamic`, chiamato da `run_dynamic`.

L'idea è iterare su tutto il sentiero temporale:

```text
input: parametri p, distribuzione iniziale F0

costruisci la griglia temporale t0,...,tN

inizializza un sentiero (F_old, V_old, controls_old)
    default: sentiero quasi-stazionario

fissa VT = V_old[N]

for m = 1,...,maxIterDynamic

    1. Calcola aggregati e prezzi dal sentiero corrente:
           K_old(t), L_old(t), LI_old(t)
           w_old(t), r_old(t)

    2. Risolvi la HJB backward:
           dato F_old(t), w_old(t), r_old(t), LI_old(t)
           e dato VT,
           calcola V_new(t) da tN a t0

    3. Ricostruisci le politiche:
           controls_new(t) = policies[V_new(t), F_old(t), prices_old(t)]

    4. Risolvi la FP forward:
           dato F0 e controls_new(t),
           calcola F_new(t) da t0 a tN

    5. Calcola nuovi prezzi dal nuovo sentiero:
           prices_new = prices[F_new, controls_new]

    6. Calcola errori:
           errF = distanza sup tra F_new e F_old
           errV = distanza sup tra V_new e V_old
           errW = distanza sup tra w_new e w_old
           errR = distanza sup tra r_new e r_old
           err  = max(errF, errV, errW, errR)

    7. Se err < tolDynamic:
           stop

    8. Altrimenti aggiorna con damping:
           F_old = (1-ωF_dynamic)F_old + ωF_dynamic F_new
           V_old = (1-ωV_dynamic)V_old + ωV_dynamic V_new
           controls_old = policies[V_old, F_old, prices(F_old)]

output: sentiero F, V, controls, prices, aggregates, diagnostics
```

### Schema Numerico Della HJB Dinamica

Per ogni data `n`, andando backward, il codice risolve una HJB implicita:

```math
\left[
\rho I + \frac{1}{\Delta t}I - A(V^n) - Q(V^n)
\right]V^n
=
u(V^n) + \frac{1}{\Delta t}V^{n+1}.
```

Qui:

- `A` è l'operatore upwind associato al drift in capitale;
- `Q` è il generatore delle transizioni epidemiologiche;
- `u` contiene flow utility e disutilità sanitarie;
- le politiche dipendono da `V^n`, quindi il sistema è risolto con un piccolo
  punto fisso locale.

Pseudocodice del passo HJB a una data:

```text
input: Vnext, guess Vn, Fn, prezzi al tempo n

for j = 1,...,maxIterHJBDynamic
    calcola ∂k Vn
    calcola c, l, q e drift b
    assembla matrice sparse della HJB implicita
    risolvi il sistema lineare per V_candidate
    residual = ||V_candidate - Vn||∞
    Vn = (1-ωHJBDynamic)Vn + ωHJBDynamic V_candidate
    se residual < tolHJBDynamic:
        break

return Vn
```

### Schema Numerico Della FP

Dato un sentiero di controlli, la FP è risolta con implicit Euler:

```math
\left(I-\Delta t\,G^n\right)\phi^{n+1}=\phi^n.
```

`G^n` è il generatore forward, composto da:

- blocchi di drift in forma conservativa e upwind;
- transizioni locali `S -> I`, `S -> R`, `I -> C`, `I -> R`, `C -> S`,
  `C -> R`, `R -> S`.

Dopo ogni passo:

1. si proietta la densità a valori non negativi;
2. si rinormalizza la massa totale a 1;
3. si salvano diagnostiche di massa e negatività numerica.

## Problema Quasi-Stazionario

Il metodo quasi-stazionario risolve una FP dipendente dal tempo, ma sostituisce
la HJB dinamica con una HJB stazionaria ricomputata lungo il tempo.

Alla data `t_n`, data la distribuzione corrente `F^n`, il solver tratta `F^n`,
`L_I^n`, `w^n` e `r^n` come condizioni correnti e risolve:

```math
\rho V_e^n(k)
=
\max_{\text{controlli}}
\left\{
\text{flow payoff}
+ b_e^n(k)\partial_k V_e^n(k)
+ \text{transizioni epidemiologiche a }t_n
\right\}.
```

Rispetto alla HJB dinamica manca il termine:

```math
\partial_t V_e(t,k).
```

Per questo il metodo quasi-stazionario non incorpora aspettative sul sentiero
futuro dell'epidemia, dei prezzi e della distribuzione. È una successione di
problemi stazionari locali.

### HJB Stazionaria

Per i suscettibili:

```math
\rho V_S(k)
=
\max_{c\ge0,\;l\in[0,1],\;q\ge0}
\Big\{
u(c,l)
- \frac{\gamma}{2}q^2
+ V'_S(k)\big[(r-\delta)k+w\eta_S l-c-\xi(t,k)q\big]
```

```math
\qquad
+ q(V_R(k)-V_S(k))
+ \beta l L_I(V_I(k)-V_S(k))
\Big\}.
```

Le equazioni per `I`, `C` e `R` sono le stesse del problema dinamico completo,
ma senza `∂t V`.

### Punto Fisso Su Wage E HJB

Dentro ogni data, la HJB stazionaria è risolta con un punto fisso annidato:

```text
input: distribuzione Ft, guess V0, wage iniziale w_start

for itw = 1,...,maxitWage

    1. Tieni fisso w.

    2. Risolvi la HJB a wage fisso:

       for itV = 1,...,maxitHJBvalue
           calcola ∂k V
           calcola lavoro, consumo, vaccino e drift
           assembla il sistema sparse:
               (ρI - A - Q)V_candidate = u
           risolvi il sistema lineare
           errV = ||V_candidate - V||∞
           V = (1-ω)V + ω V_candidate
           se errV < tolHJBvalue:
               break

    3. Con V aggiornato, calcola:
           K(Ft), L(V,Ft), w_implied(K,L), r(K,L)

    4. errW = |w_implied - w|

    5. Se errW < tolWage:
           stop

    6. Aggiorna:
           w = (1-ωw)w + ωw w_implied

output: V, w, errV, errW
```

### Loop Quasi-Stazionario Completo

Il solver `solveModel`, chiamato da `run`, usa questo schema:

```text
input: F0, parametri p

inizializza phi = F0
inizializza guess V

for n = 0,...,N-1

    1. Ft = distribuzione corrente

    2. Se è il primo passo o se scatta HJB_every:
           risolvi HJB stazionaria + wage fixed point dato Ft
       altrimenti:
           riusa le politiche precedenti

    3. Costruisci controlli e generatore FP G^n

    4. Avanza la distribuzione:
           (I - Δt G^n) phi^{n+1} = phi^n

    5. Proietta a densità non negativa e rinormalizza

    6. Salva distribuzione, value functions, controlli e prezzi

output: t, F, V, controls
```

Con `HJB_every = 1` la HJB stazionaria viene ricomputata a ogni passo della FP.
Con `HJB_every > 1` le politiche sono congelate per più passi, scelta utile solo
per accelerare esperimenti esplorativi.

## Inizializzazione Del Problema Dinamico

Il solver dinamico parte per default da:

```julia
dynamicInitialGuess = :quasistatic
```

Questo significa:

1. risolve prima il modello quasi-stazionario su tutta la griglia temporale;
2. usa il sentiero ottenuto come guess iniziale per `F(t)`, `V(t)` e controlli;
3. usa la value function quasi-stazionaria finale come terminale `V_T`;
4. avvia il punto fisso forward-backward dinamico.

Questa scelta è numericamente utile perché fornisce un sentiero già coerente con
la FP e con le restrizioni statiche dei controlli, anche se non è ancora
forward-looking.

## Discretizzazione E Stabilità Numerica

### Griglia Del Capitale

La griglia è:

```math
k_i=(i-1)\Delta k,
\qquad
i=1,\ldots,N_k,
```

con:

```math
N_k = \mathrm{Int}(MaxK/\Delta k)+1.
```

### Griglia Temporale

Il numero di passi è:

```math
N = \lceil T/\Delta t\rceil.
```

Il passo effettivo usato dal codice è:

```math
\Delta t_{eff}=T/N.
```

### Derivate Sicure

Le derivate `∂k V` sono calcolate con differenze finite:

- one-sided ai bordi;
- central all'interno;
- floor positivo `ϵDkUp`.

Questo evita divisioni per zero nelle FOC di consumo e lavoro.

### Vincoli Di Stato Sul Capitale

Il dominio del capitale è chiuso. Il codice impone vincoli di stato ai bordi:

- a `k=0`, il drift non può puntare fuori dal dominio verso sinistra;
- a `k=MaxK`, il drift non può puntare fuori dal dominio verso destra.

Operativamente:

- si corregge il consumo ai bordi;
- poi si tronca il drift:

```math
b(0)\ge 0,\qquad b(MaxK)\le 0.
```

### Operatori Sparse

Sia nella HJB sia nella FP, gli operatori sono assemblati come matrici sparse.
Il drift in capitale usa uno schema upwind di primo ordine.

Nella HJB si risolve un sistema lineare implicito per le value functions.
Nella FP si risolve un sistema lineare implicito per la densità.

## Output Del Solver Dinamico

`solveModelDynamic` restituisce un `NamedTuple` con:

```text
t
    griglia temporale salvata

F
    sentiero delle distribuzioni

V
    sentiero delle value functions, con w, r, LI associati

controls
    consumi, lavoro, vaccino, drift, intensità di transizione

prices
    w, r, K, L, LI

aggregates
    K, L, LI

diagnostics
    err, errF, errV, errW, errR,
    errori di massa, minimi di densità,
    diagnostiche HJB e FP

converged
    booleano di convergenza

iterations
    numero di iterazioni Picard dinamiche

method
    :forward_backward_dynamic
```

## Figure

La funzione:

```julia
EpiEconMFG.save_all_figures(result, p; outdir = "outputs/figures")
```

genera figure su:

- masse aggregate `S`, `I`, `C`, `R` nel tempo;
- distribuzioni `φ_e(t,k)`;
- quote relative per stato;
- flussi `S -> I` e `S -> R`;
- consumi;
- lavoro;
- intensità vaccinale `q`;
- wage effettivo dei suscettibili;
- `R0` implicito;
- distribuzione totale della ricchezza.

Con:

```julia
with_surfaces = true
```

salva anche superfici 3D in PNG oltre alle heatmap/PDF.

## Parametri Principali

I parametri sono definiti in `src/core/parameters.jl` dentro `MFGEpiEcon`.

Alcuni parametri numerici importanti:

```text
T_End
    orizzonte temporale

MaxK, Δk
    dominio e passo del capitale

Δt
    passo temporale desiderato

ξ
    costo monetario esogeno del vaccino

tolDynamic, maxIterDynamic
    tolleranza e massimo numero di iterazioni del punto fisso dinamico

ωF_dynamic, ωV_dynamic
    damping sul sentiero della distribuzione e delle value functions

maxIterHJBDynamic, tolHJBDynamic, ωHJBDynamic
    controlli del punto fisso locale nella HJB dinamica

tolHJBvalue, maxitHJBvalue, ω
    controlli della HJB stazionaria

tolWage, maxitWage, ωw
    controlli del punto fisso sul wage nel quasi-stazionario
```

## Stato Corrente Del Codice

Il problema dinamico completo è implementato e viene risolto da
`solveModelDynamic`. Il metodo quasi-stazionario rimane nel codice perché è
ancora utile per:

- costruire il guess iniziale del solver dinamico;
- confrontare la soluzione full dynamic con la soluzione non-forward-looking;
- fare debug separato di HJB stazionaria e FP.

La convenzione corrente è lavorare sul solver dinamico come oggetto principale e
usare il quasi-stazionario solo come strumento ausiliario.
