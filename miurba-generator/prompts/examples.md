## generation
EJEMPLOS DE EXPRESIÓN CORRECTA E INCORRECTA (el escenario manda; no inventes):

- Urgencia — expresiones permitidas, por nivel (usa SOLO las del nivel asignado):
{{urgency_examples}}
  INCORRECTO: escenario con gravedad "alta" y el texto dice "¡Es una emergencia, vengan ya mismo!" (usa expresiones de inmediata).
  CORRECTO: escenario "alta" → "Es importante y me preocupa, deberían revisarlo pronto."

- Urgencia — expresiones PROHIBIDAS en este registro (pertenecen a niveles que no se han pedido):
{{prohibited_block}}
  Si necesitas cerrar la queja, usa una expresión del nivel asignado, no de estos.

- Elementos afectados — son partes del mismo problema, nunca problemas nuevos:
  INCORRECTO: "Hay una fuga en la tubería. Además hay una avería en el canalón. Y el muro está roto." (tres problemas inventados)
  CORRECTO: "Hay una fuga en la tubería del portal; además se ha afectado el canalón y el muro cortafuegos está dañado."

- Riesgos — se nombran con su tipo de daño:
  INCORRECTO: "esto es peligrosísimo, no se puede vivir así" (miedo genérico: no cuenta como riesgo)
  CORRECTO: "hay riesgo eléctrico porque los cables quedan al descubierto"

- Ubicación — zona y detalle concretos:
  INCORRECTO: "por ahí arriba, en la zona común"
  CORRECTO: "en la cubierta, en la zona exterior, junto al canalón del patio de luces"

- Categoría y subcategoría — deben reconocerse en el texto:
  INCORRECTO: "hay un problema de mantenimiento en el edificio" (no se reconoce el tipo de problema)
  CORRECTO: "hay fugas de agua en la tubería" (se reconoce fugas y tubería)

- Timestamp: ISO 8601 realista, por ejemplo "2024-03-15T11:23:47+0000".

## judge
EJEMPLOS RESUELTOS (no son el caso a evaluar; muestran el criterio):

EJEMPLO 1 — registro APTO
  ESCENARIO: 1 problema: fontanería/fugas, zona portal (hall de entrada), gravedad media, elementos [tubería/vertical], riesgos [ninguno].
  TEXTO: "Buenas, en el hall de entrada hay una fuga de agua en la tubería vertical. No tiene prisa pero me preocupa el gasto. Cuando puedan, gracias."
  RAZONAMIENTO: categoría y subcategoría coinciden; la expresión "no tiene prisa pero" es de media, como el escenario; nombra la tubería; no nombra riesgos (el escenario dice ninguno); 1 problema.
  SALIDA: {"fields": {"category": "ok", "subcategory": "ok", "urgency": "ok", "affected_elements": "ok", "risks": "ok", "location": "ok"}, "n_problems_texto": 1, "urgencias_detectadas": ["media"], "expresion_citada": "no tiene prisa pero", "inventado": {"elementos": [], "riesgos": [], "problemas": []}, "omitido": {"elementos": [], "riesgos": []}, "evidencia": "", "valid": true, "issues": []}

EJEMPLO 2 — registro NO APTO por urgencia y por elemento inventado
  ESCENARIO: el mismo anterior.
  TEXTO: "¡Es una emergencia, vengan ya mismo! La tubería vertical pierde y además se ha dañado el muro cortafuegos."
  RAZONAMIENTO: "es una emergencia" es de inmediata y el escenario pide media → urgency grave; el muro cortafuegos no está en el escenario → affected_elements grave e inventado; 1 problema.
  SALIDA: {"fields": {"category": "ok", "subcategory": "ok", "urgency": "grave", "affected_elements": "grave", "risks": "ok", "location": "ok"}, "n_problems_texto": 1, "urgencias_detectadas": ["inmediata"], "expresion_citada": "es una emergencia", "inventado": {"elementos": ["muro/cortafuegos"], "riesgos": [], "problemas": []}, "omitido": {"elementos": [], "riesgos": []}, "evidencia": "es una emergencia / muro cortafuegos", "valid": false, "issues": ["urgency", "affected_elements"]}

EJEMPLO 3 — registro APTO con dos problemas
  ESCENARIO: 2 problemas: (1) fontanería/fugas, garaje nivel -1, media, [tubería/horizontal], [inundación]; (2) limpieza/basuras, portal zona de buzones, baja, [suelo/solería], [ninguno].
  TEXTO: "En el garaje, nivel -1, hay una fuga en la tubería horizontal; no tiene prisa pero me preocupa que se inunde. Y en el portal, junto a los buzones, hay basura acumulada en el suelo: es una sugerencia, para cuando haya tiempo."
  RAZONAMIENTO: dos problemas, en el orden del escenario; el primero con "no tiene prisa pero" (media) y riesgo de inundación; el segundo con "es una sugerencia" (baja) y sin riesgos.
  SALIDA: {"fields": {"category": "ok", "subcategory": "ok", "urgency": "ok", "affected_elements": "ok", "risks": "ok", "location": "ok"}, "n_problems_texto": 2, "urgencias_detectadas": ["media", "baja"], "expresion_citada": "no tiene prisa pero / es una sugerencia", "inventado": {"elementos": [], "riesgos": [], "problemas": []}, "omitido": {"elementos": [], "riesgos": []}, "evidencia": "", "valid": true, "issues": []}

CRITERIO DE VEREDICTOS
- ok: el campo se corresponde con el escenario (sinónimos y lenguaje coloquial permitidos).
- menor: el objeto se nombra pero falta su variedad concreta, o el matiz es impreciso sin contradecir el escenario.
- grave: el campo falta, se contradice o se inventa algo que el escenario no tiene.
