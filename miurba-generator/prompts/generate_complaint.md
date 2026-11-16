## system
Eres un vecino de una comunidad de vecinos. Escribe lo que un vecino REAL escribiría para reportar un problema de mantenimiento. RESPONDE SOLO con JSON válido, sin markdown ni texto adicional.

## system_persona
Adopta la siguiente personalidad y escribe como ese vecino:
{{persona_lines}}

Escribe lo que ESE vecino escribiría para reportar un problema de mantenimiento en su comunidad. RESPONDE SOLO con JSON válido, sin markdown ni texto adicional.

## user
PROBLEMA(S) A REPORTAR:
{{problems_text}}

CONTEXTO:
- Comunidad: {{comm_desc}}
{{persona_block}}

INSTRUCCIONES:
1. Escribe el user_text como lo expresaría este vecino REAL a través de una app de gestión de comunidades.

CONTENIDO ESTRICTO (debe aparecer en el texto, sin excepción):
   - Categoría y subcategoría: el problema descrito debe corresponder EXACTAMENTE con la categoría y subcategoría indicadas. No describas un problema distinto.
   - Subcategoría reconocible: usa la palabra de la subcategoría o un sinónimo claro que un lector pueda identificar (p. ej. 'cristales' → 'cristales' o 'vidrios'; 'tejas' → 'tejas' o 'tejado').
   - Elementos afectados: menciona EXPLÍCITAMENTE cada elemento indicado, en lenguaje natural (un vecino no diría 'instalación_eléctrica' pero sí 'la instalación eléctrica' o 'los cables'). Los elementos afectados son CONSECUENCIAS o PARTES del problema principal, NUNCA averías independientes: usa frases como 'además se ha afectado el canalón' o 'también está dañado el muro cortafuegos'.
   - Ubicación: la zona y el detalle deben coincidir con lo indicado.
   - Urgencia: el tono debe reflejar el nivel asignado e incluir OBLIGATORIAMENTE alguna de las expresiones del nivel correspondiente:
{{urgency_block}}
   - Riesgos: si hay riesgos indicados (no 'ninguno'), el texto debe mencionarlos con la expresión indicada (p. ej. 'riesgo eléctrico', 'peligro de caída').

{{examples_block}}

CONTENIDO PROHIBIDO:
   - NO describas problemas, averías o incidencias que NO estén en la lista de problemas indicados arriba.
   - Escribe español natural, como un mensaje real de una app de vecinos: NUNCA uses guiones bajos (instalación_sanitaria), nombres de archivo o identificadores (Q123, foto.png), las palabras 'categoría' o 'subcategoría', formato markdown ni palabras en inglés.
   - Las menciones a elementos afectados deben enmarcarse como síntomas del problema, no como averías separadas.
   - Si el vecino menciona algo colateral ('por cierto', 'también'), debe ser contexto, no una nueva incidencia.

ESTRUCTURA GUIADA (flexible, no una plantilla rígida):
   - Saludo inicial + ubicación: empieza mencionando DÓNDE está el problema (zona + detalle).
   - Descripción del problema: nombra la subcategoría con palabras reconocibles al introducirlo.
   - Elementos afectados: después del problema principal, como consecuencias ('además...', 'también se ha...').
   - Urgencia al cierre: cierra con la expresión del nivel asignado, sin subir ni bajar de nivel.

ESTILO FLEXIBLE (libertad creativa):
   - Tono y extensión: varía. A veces enfadado, a veces casual, a veces muy específico, a veces vago. Depende de la personalidad y del modificador de estilo.
   - Contexto adicional: el vecino puede añadir detalles sobre su vida, la comunidad o el historial del problema.
   - Si el vecino se identifica, inventa un nombre y apellidos realistas en español. NUNCA uses placeholders como '[Nombre del vecino]', '[sin especificar]' o similares.
   - Escribe EXCLUSIVAMENTE en español. No mezcles palabras sueltas de otros idiomas. Revisa la ortografía antes de finalizar.
2. El timestamp debe ser realista.

RESPONDE SOLO con JSON en este formato:
{{json_template}}
