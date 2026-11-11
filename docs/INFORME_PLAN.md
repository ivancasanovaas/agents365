# Informe de situación y plan de ataque: More than a Hack 2027, Fase 1 (posesión, BAR-ATH)

Fecha: 2026-11-11 · **Deadline: 2026-11-17 (quedan 6 días)** · Estado: exploración de datos (notebooks 01-03) y brainstorming de mejoras.

> Este informe se ha construido solo con la documentación del organizador, el código de `src/` y `evaluation/scoreboard.csv`.
> **No se ha leído nada de `data/`** (regla dura). Todo lo que requiere ver datos va como script/pregunta para que lo ejecutéis vosotros en local y miréis **solo agregados**.

---

## 1. Resumen ejecutivo

| | |
|---|---|
| Tarea | Una etiqueta por frame (199.319 frames, 25 fps): `H` / `A` / `D` / `X` |
| Métrica de ranking | **spellF1** (por posesión, IoU temporal 0.3-0.7). Desempate: macroF1 |
| Suelo a batir | baseline_smooth50 = **spellF1 0.2433** (macroF1 0.6724) |
| Ruido | spellF1: diferencias < ~0.04 son ruido. macroF1: < ~0.01 |
| Dato clave | El balón falta en **41 %** de los frames puntuados; el tracking es automático, sin identidades, 46 slots para 22 jugadores |
| Lo que ya tenemos | Pipeline HMM de 3 estados + Viterbi (`src/model/hmm.py`, `src/run.py`), banco de validación externo (Metrica G3, SkillCorner), degradador, guardrails, borrador de informe |
| Resultado actual (solo externo) | HMM 0.41 / 0.40 spellF1 frente a 0.26 / 0.31 del baseline. **Nunca ejecutado sobre el partido real** |
| Mayor hueco | Detección de **balón muerto (D)** casi inexistente; Smart Tagging, homografía y vídeo sin usar |

**Tesis del plan:** spellF1 premia acertar *intervalos de posesión* (cuántas, dónde empiezan y acaban), no frames. Las ganancias grandes vienen de (1) no fragmentar, (2) detectar bien los parones (D parte posesiones) y (3) acertar los cambios de posesión. Lo demás es secundario.

---

## 2. Qué es exactamente lo que se puntúa (lo imprescindible)

1. **Posesión = racha máxima de H o de A con balón vivo.** `H H D D H H` son **dos** posesiones. Un `D` mal puesto en medio de una posesión la parte en dos: el baseline "sin balón → D" fragmenta masivamente (131 posesiones frente a 17 en los primeros 5 min).
2. Emparejamiento 1-a-1 por IoU temporal, F1 medio en umbrales 0.3…0.7. Un desfase de 1 s ya baja a 0.65; de 8 frames, 0.89. **La precisión del borde importa**, pero ±3 frames casi no cuesta.
3. **`X` en spellF1 = "repite mi última etiqueta de esa mitad".** Dentro de una posesión es gratis; en un relevo es caro. En macroF1 cada `X` puntuado es un fallo. Conclusión: salvo fuera de la zona puntuada, **no abstenerse** (ya lo hace nuestro pipeline; EXP-001 lo confirma en diseño).
4. **Zona puntuada:** frames 11.185-90.651 y 110.808-190.197. Trampa: 110.808-113.741 (2.934 frames) están puntuados y son todos `D` (ya forzado en `src/competition.py`).
5. **D de dueño:** nunca se pide equipo en balón muerto; es solo `D`.
6. La definición de "control" (no un simple toque, no un desvío, no un duelo aéreo sin control posterior) la fija el GT público: **sirve para entender la semántica, no para entrenar ni afinar**.

---

## 3. Reglas que condicionan todo (checklist de cumplimiento)

- Nunca leer/imprimir contenido de `data/` desde el agente; scripts que tocan datos imprimen **agregados**.
- Prohibido anotar o corregir a mano este partido; prohibidas reglas atadas a frames concretos.
- El GT público (5 min, 17 posesiones) **solo comprueba que el pipeline es correcto**; no entrena, no ajusta umbrales, no elige entre variantes (demostrado: elige mal incluso en un baseline trivial).
- Todo umbral/constante con origen documentado en `PROVENANCE.md` (FIFA Laws, física, datos abiertos de otros partidos, estimación no supervisada sin etiquetas humanas).
- Datos externos: solo de **otros partidos**, accesibles por el organizador y documentados. Prohibido cualquier dato humano (eventos, tracking corregido…) de BAR-ATH.
- LLMs en la nube: no pueden recibir nada de los datos de la competición ni extractos (T&C).
- Repo público, sin datos ni exports frame-level; **un comando** regenera el CSV idéntico.
- Entrega: `<team_name>.csv` + `<team_name>.pdf` (o .md) + URL del repo, vía Google Form (cuenta Google necesaria). Equipos de 3-5 personas.

---

## 4. Qué hay en los datos y qué preguntas responder en los notebooks 01-03

Principio: **mirar para entender y diseñar features** está permitido; convertir lo visto en etiquetas, correcciones o reglas atadas a frames, no. Pegad solo agregados al chat si necesitáis ayuda.

### 4.1 `01_smart_tagging.ipynb`: 638 fases / jugadas a balón parado

Qué es: anotación automática de Metrica (intervalos con `code`, `Team`, `Half`, `Type`, `Side`…). `Team` = Barcelona 314, Athletic 288, **N/A 36 (incluye todos los saques de banda)**. Tipos de balón parado: saque de banda 26, córner 12, saque de puerta 10, falta 5, saque inicial 6.

Preguntas a resolver (todas con agregados):
- **Cobertura temporal:** ¿qué % de los frames puntuados cae dentro de algún intervalo? ¿Hay huecos largos sin etiquetar? (los huecos suelen ser D o transiciones.)
- **Los `SET PIECES` (59):** ¿el intervalo empieza en el parón o en el restart? Distribución de duración (mediana, p10, p90). Esto decide si sirve como **detector de D** (la clave).
- **Relación con el modelo:** una vez tengáis la salida del HMM, calcular **concordancia** (% de frames en intervalo `Team=Barcelona` donde el HMM dice H, y viceversa). Es una métrica de consistencia interna, **no** contra GT.
- ¿Cuántos intervalos solapan entre sí? ¿Con qué frecuencia dos fases consecutivas tienen distinto `Team`? (proxy de cambio de posesión → ¿coincide con los cambios del HMM dentro de ±2 s?)
- Esperanza realista: sirve como **prior suave**, no como etiqueta. Los intervalos son fases, no posesiones.

### 4.2 `02_tracking_atd.ipynb`: el tracking automático

Preguntas a resolver:
- **Cobertura del balón por minuto** (ya sale en 03) y **por zona del campo**: ¿dónde falta más (áreas, bandas, juego aéreo)? ¿Longitud de las rachas de ausencia (histograma: mediana, p90, máximo)? Esto dimensiona cuánto "puente" tiene que aguantar el HMM.
- **Calidad de los slots:** nº medio de objetos por frame (documentado 20.5), frecuencia de >22 (duplicados), nº de cambios de equipo/identidad aparentes por minuto.
- **Física del balón visible:** distribución de velocidad (m/s). Saltos imposibles (> ~40 m/s, regla física) = detecciones espurias → candidatos a filtro (umbral con origen físico en PROVENANCE).
- **Balón fuera del campo:** % de frames con balón con coordenadas fuera de [0,1]². Es el primer detector gratis de D (líneas de banda/fondo, Ley 9 FIFA).
- **Parones colectivos:** ¿hay momentos donde casi todos los jugadores están casi quietos / agrupados (saques, faltas, celebraciones)? Cuantificar con velocidad media del grupo, sin ver frames sueltos.
- **Sentido de ataque:** posición media de cada equipo por mitad (el equipo local cambia de lado en la 2ª parte); útil como feature "balón en campo propio/rival".

### 4.3 `03_first_submission.ipynb`: baseline + scorer

- Reproducir los 4 baselines y comprobar que el scorer devuelve las cifras de `scoreboard.csv` (sanidad de pipeline).
- Mirar el **recuento de posesiones** de cada variante frente a 17 (aprender por qué raw ≈ 131).
- Ejecutar nuestro `python -m src.run` en la DGX Spark y revisar **solo**: etiquetas por clase, nº de cambios H↔A, nº de posesiones, duración media/mediana, validación OK.
- **Cheque de cordura sin GT** (permitido, es consistencia, no tuning): ¿la duración media de posesión es del orden de 10-30 s? ¿D representa una fracción razonable del tiempo (el balón está vivo ~55-60 min de ~95)? Si D≈0 %, falta el detector de balón muerto.

---

## 5. Diagnóstico del pipeline actual (qué hace y dónde está débil)

Evidencia por frame: equipo del jugador más cercano al balón (5 cubos de distancia). Sin balón no hay evidencia y la matriz de transición arrastra el estado. Parámetros por conteo en Metrica G3; `TEMP = 0.3`; velocidad del balón desactivada porque no transfería.

| Debilidad | Efecto probable en spellF1 |
|---|---|
| **D casi no se detecta** (solo bloque pre-saque 2ª parte) | Posesiones largas fusionadas a través de parones → menos spells, bordes erróneos. Mayor palanca pendiente. |
| "Más cercano al balón" confunde posesión con proximidad (duelos, pases largos, balón en el aire) | Cambios espurios H↔A en disputas. |
| Sin duración explícita | Spells de 1-2 s sobreviven. Una posesión real dura segundos (media GT público: 14.2 s). |
| Balón ausente 41 %: todo se apoya en la transición | Relevos durante huecos se detectan tarde o nunca. |
| Parámetros de 1 partido externo; `temp` elegido sobre 2 partidos | Resultados optimistas. La ganancia se reduce a la mitad al transferir y macroF1 baja (0.70→0.59, 0.73→0.54). |
| No usa Smart Tagging, homografía, vídeo, tracking broadcast | Capital sin explotar. |

---

## 6. Brainstorming de mejoras, priorizado

Escala: **Impacto** (en spellF1 esperado) · **Coste** (horas) · **Riesgo** (de incumplir reglas o de no transferir).

### Prioridad A: probablemente grandes ganancias, coste razonable

1. **Detector de balón muerto (D) algorítmico.** Combinar:
   - balón fuera de límites (líneas/fondos; Ley 9);
   - balón ausente + jugadores casi quietos / agrupados (restart inminente);
   - inicio/fin por `SET PIECES` de Smart Tagging (tras verificar en agregados su alineación con parones);
   - balón casi estacionario (velocidad < umbral físico) durante > N s;
   - distancia mínima 9.15 m en faltas/córners (Ley 13/17) como señal de barrera.
   Se integra como **cuarta emisión** en el HMM (o como un estado D con emisión propia) en lugar de reglas duras.
   Impacto: alto · Coste: medio · Riesgo: bajo si cada umbral va a PROVENANCE.
2. **Duración explícita de posesión (semi-Markov / HSMM) o filtro de duración mínima.** Aprender la distribución de duraciones de posesión **en partidos externos** (no en BAR-ATH) y penalizar spells < ~2 s. Alternativa barata: Viterbi con matriz de transición más "pegajosa" + post-proceso que fusiona spells cortos entre dos del mismo equipo. Impacto: medio-alto · Coste: bajo-medio.
3. **Proxy de "control", no de proximidad.** Pasar de "jugador más cercano" a "equipo con control sostenido": jugador del mismo equipo a < ~2 m durante ≥ ~1-2 s (regla de CHALLENGE.md: "un jugador mantiene el balón unos segundos" / ">2 pases consecutivos"), y cambiar de posesión solo cuando el rival lo logra. Se implementa como evidencia agregada en ventana (suma de proximidad ponderada por equipo) en lugar de un único frame. Impacto: alto · Coste: medio.
4. **Imputación del balón en huecos.** Interpolación física (velocidad constante/Kalman) para huecos cortos, y para huecos largos usar el **centroide del equipo en posesión / cluster de jugadores** como balón aproximado. Evita que 41 % de frames queden "ciegos". Impacto: medio-alto · Coste: medio.

### Prioridad B: ganancia moderada, útil para robustez

5. **Prior de Smart Tagging como evidencia suave** (`Team` en intervalos no N/A) con peso bajo y documentado; tratar `N/A` explícitamente (los saques de banda: equipo = el que *no* tocó último, o simplemente sin evidencia). Se valida con concordancia interna, no con el GT.
6. **Features de contexto espacial:** campo propio/rival según sentido de ataque, distancia a línea de gol, nº de jugadores de cada equipo cerca del balón (superioridad local). Un clasificador (regresión logística / GBM pequeño) entrenado sobre partidos externos degradados.
7. **Más datos externos para entrenar:** Metrica Sample Games 1-2 (abiertos), más partidos SkillCorner (MIT). Entrenar conjuntamente y validar con **leave-one-game-out**. Es la mejor defensa contra el sobreajuste de `temp` y buckets. Documentar cada uno.
8. **Decodificación afinada para IoU:** el emparejamiento premia el solape; corregir el **sesgo de latencia** de los bordes (si el modelo detecta sistemáticamente los cambios con retraso en los partidos externos, adelantar la frontera una cantidad estimada ahí, no aquí).

### Prioridad C: alto coste / alto riesgo; solo si sobra tiempo

9. **Sincronizar el tracking broadcast** con el táctico (correlación cruzada de trayectorias de jugadores/balón, algorítmica) para rellenar huecos del balón con la segunda cámara (50.5 % visible; parte complementaria). Complejo por cortes y repeticiones (34 % sin homografía válida).
10. **Detector de balón propio sobre vídeo / re-tracking.** Legítimo por reglas, pero coste de GPU y tiempo enorme para 6 días.
11. **Modelo secuencial neuronal** (BiLSTM/Transformer) entrenado sobre externos degradados. Posible, pero el HMM interpretable puede dar el 80 % con mucho menos riesgo de transferencia.

### Lo que no hay que hacer

- Elegir variantes, ventanas o umbrales mirando la puntuación del GT público.
- Reglas del tipo "en el minuto X hay un córner".
- Subir cualquier extracto de los datos a un LLM en la nube (esto incluye pegar filas en el chat).
- Usar `X` dentro de la zona puntuada como "no sé": cuesta macroF1 y spellF1 solo se salva dentro de posesiones.

---

## 7. Metodología para decidir sin trampa

1. **Banco externo con degradación realista** (`src/external/degrade.py`: balón ausente 41 %, 20.5 jugadores, swaps de identidad, ruido). Los parámetros no publicados (longitud de huecos, tasa de swaps…) están marcados como SUPUESTOS: **barrerlos**, no ajustarlos.
2. **Regla de decisión fijada de antemano:** una variante gana solo si mejora spellF1 en **> 0.04** de forma consistente entre semillas y partidos, con `--bootstrap 1000`. Si no, se queda la más simple.
3. **Leave-one-game-out** con ≥ 3 partidos externos (hoy hay 2).
4. **Métricas a vigilar a la vez:** spellF1, macroF1, nº de posesiones frente a las del partido de referencia, fragmentación, latencia media/mediana de los bordes y D-F1.
5. **Sanidad en BAR-ATH** (solo agregados, sin ajustar): validación OK, recuento de posesiones y fracción de D plausibles, determinismo (SHA-256 de dos ejecuciones iguales).
6. Cada constante nueva → fila en `PROVENANCE.md` **antes** de merge (usar skill `threshold-provenance`).

---

## 8. Plan de 6 días (11 → 17 de noviembre)

| Día | Objetivo | Entregable |
|---|---|---|
| **Mié 11** | Cerrar entendimiento de datos (notebooks 01-03 con agregados); ejecutar `python -m src.run` en la DGX y revisar agregados | Notas de EDA (solo números), primera lista de rachas de balón ausente, % D, nº posesiones |
| **Jue 12** | Detector de D v1 (límites del campo + quietud + `SET PIECES` si procede) integrado como emisión; filtro de duración mínima | EXP-003 definido y ejecutado en banco externo |
| **Vie 13** | Proxy de control sostenido + imputación de balón en huecos cortos | EXP-004, comparación con bootstrap |
| **Sáb 14** | Más partidos externos (Metrica 1-2, SkillCorner extra) y leave-one-game-out; fijar `temp`/buckets de forma no optimista | `models/hmm_params.json` regenerado con un comando, PROVENANCE al día |
| **Dom 15** | **Congelar el método.** Ejecución final en la DGX, validación (`validate_submission.py`), determinismo (2 ejecuciones, mismo hash) | `outputs/submission.csv` válido |
| **Lun 16** | Informe técnico (skill `tech-report`), auditoría de cumplimiento (`compliance-auditor`), limpiar repo (sin datos, `requirements.txt` pinned), nombrar `<team_name>.csv/.pdf` | Paquete de entrega completo |
| **Mar 17** | Colchón: subir el formulario con margen (cuenta Google, repo público accesible); si falla algo, reproducir | Entregado antes del cierre |

Regla de oro: **a partir del domingo no se añaden características**; solo se arregla lo que rompa validación o reproducibilidad.

---

## 9. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| El HMM nunca se ha corrido sobre BAR-ATH → parser o formato inesperado | Ejecutarlo hoy; es lo primero. Mirar solo agregados |
| La ganancia externa (0.41) no transfiere al partido real (ya se redujo a la mitad entre partidos) | Más partidos externos + leave-one-game-out; no sobreprometer en el informe |
| Definición de D del GT (tracking corregido) distinta a la derivada de eventos | Documentar la diferencia (ya está en TECH_REPORT §5); basar D en física y en reglas FIFA, no en eventos |
| Sobreajuste al GT público de 17 posesiones | Prohibido tunear; usarlo solo como verificación tras congelar |
| Incumplimiento accidental (pegar datos en el chat, commit de CSV por frame) | Hook `block_data_access.py`, `.gitignore`, auditor antes de cada commit |
| No determinismo / no reproducibilidad | Semillas fijas, dependencias pinned, comparar SHA-256 de dos ejecuciones |
| Subida de última hora fallida | Entregar el día 16, no el 17 |

---

## 10. Entregables y estado

| Entregable | Estado | Pendiente |
|---|---|---|
| CSV `frame,label`, 199.319 filas | Pipeline listo, no ejecutado sobre el partido real | Ejecutar, validar, congelar |
| Repo público | Estructura y guardrails listos | Pinear dependencias, README de ejecución, sin datos |
| Informe técnico | Borrador en `docs/TECH_REPORT.md` | Rellenar TODOs (resultado real agregado, conteos finales, hash), nombre del equipo, PDF |

---

## 11. Preguntas abiertas para el equipo

1. ¿Quién ejecuta lo que toca `data/` en la DGX y devuelve **solo** agregados?
2. ¿Qué cobertura temporal tienen realmente los `SET PIECES` respecto a los parones? (decide el valor del detector de D).
3. ¿Cuántos partidos externos adicionales podemos conseguir en 2 días (Metrica 1-2, SkillCorner)?
4. ¿Nombre del equipo definitivo (nombra el CSV y el PDF)?
5. ¿Intentamos la sincronización broadcast o se descarta explícitamente y se documenta como "probado y descartado/no abordado"?
