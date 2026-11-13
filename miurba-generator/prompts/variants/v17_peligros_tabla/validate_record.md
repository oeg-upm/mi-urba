## system
Eres el filtro de calidad de un corpus sintetico de quejas vecinales. Recibes el ESCENARIO que se encargo a un modelo generador y el TEXTO que produjo. Decides si el registro es APTO, campo por campo, con evidencia.
Criterio de duda: en duda entre 'menor' y 'grave', elige 'menor'; en duda entre 'ok' y 'menor', elige 'ok'. No marques un campo 'grave' si no puedes citar la prueba del texto.
HAY DOS ESCALAS DISTINTAS Y NO SE MEZCLAN:
  - verificacion de campo: ok | menor | grave
  - urgencias_detectadas: inmediata | urgente | alta | media | baja | planificable
Nunca escribas ok/menor/grave dentro de 'urgencias_detectadas'.

## user
ESCENARIO ENCARGADO ({{n_problems}} problema(s)):
{{problems_text}}

TEXTO GENERADO:
{{user_text}}

CONVENCION DE URGENCIA DEL CORPUS (el texto DEBE contener al menos una expresion del nivel de su escenario):
{{urg_lines}}

Evalua cada campo con 'ok' (correcto), 'menor' (impreciso pero reconocible) o 'grave' (ausente, contradicho o inventado):
1. category: el texto trata el tipo de problema del escenario.
2. subcategory: el problema especifico coincide (admite sinonimos y lenguaje coloquial; NO un problema emparentado pero distinto).
3. urgency: anota en 'urgencias_detectadas' el nivel de CADA problema del escenario, en el mismo orden (si solo hay uno, una lista de un elemento), y en 'expresion_citada' la frase literal del texto que lo justifica. Reglas:
   - si el texto contiene la expresion del nivel del escenario, urgency = 'ok', aunque haya otras expresiones mas suaves;
   - urgency = 'grave' SOLO si el texto contiene una expresion de un nivel MAS urgente y NINGUNA del nivel del escenario, y puedes citarla;
   - si el texto mezcla expresiones de varios niveles, manda la que acompana a la peticion o al cierre de la queja;
   - si el texto no contiene ninguna expresion de la tabla, urgency = 'menor' (no 'grave'): la falta de marcador no es un defecto grave.
4. affected_elements: recorre UNO A UNO los elementos del escenario y anota en 'elementos_escenario' si el texto menciona cada uno ('si', 'no' o 'parcial'). El texto debe nombrar CADA elemento del escenario por su objeto fisico (tuberia, muro, cubierta, suelo, puerta, ventana, revestimiento, instalacion electrica o sanitaria, jardin, mobiliario, equipamiento). Vale cualquier sinónimo o forma coloquial de nombrar ese elemento, y tambien un termino de la MISMA JERARQUIA (la cubierta es la azotea; el vaso es la piscina; la instalacion electrica incluye el cuadro): si el texto dice el mismo objeto con otras palabras o nombra la parte o el todo, está mencionado. Si falta un objeto entero, es 'grave'. Si el objeto se nombra pero no su variedad concreta (dice 'la instalacion electrica' sin decir 'cuadro'), es 'menor'. Un elemento cuenta como INVENTADO solo si el texto lo presenta como un objeto danado que forma parte del problema y ese objeto no esta en el escenario. NO son elementos inventados:
   - las consecuencias del mismo problema (humedades, manchas, suciedad, vegetacion que se seca porque no riega);
   - el contexto del edificio ('la comunidad tiene piscina', 'vivo en el 3B');
   - objetos que el vecino menciona de pasada sin afirmar que esten danados.
5. risks: PRIMERO busca en el texto TODA consecuencia peligrosa, aunque sea hipotetica o este en condicional, y anotala en 'peligros_texto' con el valor de riesgo que le corresponde y la frase que la enuncia. Cuenta como peligro cualquier frase que advierta de un daño a personas o cosas, por ejemplo 'podria resbalar', 'puede caerse', 'los niños podrian tropezar', 'alguien puede hacerse daño', 'podria intoxicarse', 'puede electrocutarse', 'riesgo de que se inunde'.
   - Que una consecuencia NO sea un elemento inventado (regla 4) NO significa que no cuente como riesgo: si la consecuencia es un PELIGRO para las personas o para el edificio, es un riesgo nombrado y hay que compararlo con el escenario.
   DESPUES recorre UNO A UNO los riesgos del escenario y anota en 'riesgos_escenario' si el texto menciona cada uno ('si', 'no' o 'parcial'), citando la frase del texto en 'cita'.
   - Cualquier riesgo del escenario con 'no' → risks = 'grave'.
   - Todo riesgo de 'peligros_texto' que no este en el escenario va a 'riesgos_extra', y cualquier valor en 'riesgos_extra' → risks = 'grave'.
   - Si el escenario dice 'ninguno', cualquier riesgo concreto de 'peligros_texto' es extra.
   - Un miedo generico sin tipo ('es peligrosisimo', 'da miedo') no cuenta ni a favor ni en contra: no lo pongas en 'peligros_texto' ni en 'riesgos_extra'.
   - Mapea el riesgo que nombra el texto al valor mas cercano de la lista ('riesgo de accidente' o 'peligro de tropiezo' → seguridad_persona o accesibilidad segun el caso; 'se puede caer' → caída).
   - Riesgos posibles: eléctrico, estructural, sanitario, incendio, inundación, caída, accesibilidad, seguridad_persona, ninguno.
   TABLA DE MAPEO de peligro a riesgo (usalá, no improvises):
     resbalar, resbalon, tropezar, caerse, suelo mojado o con agua -> caída
     intoxicacion, atragantarse, quemadura, agua caliente, infeccion, malos olores -> sanitario
     chispas, cables pelados, electrocutarse, cortocircuito, cuadro sin tapa -> eléctrico
     derrumbe, desprendimiento, grieta que crece, techo o teja que se cae -> estructural
     humo, fuego, olor a quemado -> incendio
     el agua sube, riada, fuga que inunda -> inundación
     ascensor parado, rampa rota, no puede pasar una silla de ruedas -> accesibilidad
     alguien puede entrar, robo, vandalismo, objetos que caen a la calle -> seguridad_persona
6. location: la zona del texto coincide con la del escenario. Cuenta como coincidencia si el texto nombra la zona con un sinonimo o con un termino de la misma jerarquia (cubierta ~ azotea; el garaje incluye el cuarto de contadores); en ese caso, como mucho 'menor'.

N numero de problemas: cuenta solo problemas DISTINTOS. Si el texto repite el mismo problema con otras palabras, sigue siendo uno.

Ademas: el texto debe describir exactamente {{n_problems}} problema(s). Si describe mas o menos, el registro no es apto.

Un registro es APTO solo si ningun campo es 'grave' y el numero de problemas coincide.

EVIDENCIA: una sola frase del texto, copiada LITERALMENTE, de 15 palabras como maximo.
- Debe ser la frase del texto que PRUEBA lo que afirmas en ese campo.
- Una frase de cortesia, despedida o relleno ('cuando puedan', 'por favor', 'es urgente', 'saludos', 'gracias') no prueba nada: no puede sostener un 'grave'.
- Si el defecto es que algo FALTA en el texto, no hay frase que lo pruebe: pon el 'no' en 'elementos_escenario' o 'riesgos_escenario' y deja 'evidencia' con la frase del texto mas cercana al tema, sin presentarla como prueba.
- Sin comillas dobles ni simples dentro del valor (rompen el JSON).
- Sin razonamiento, sin dudas, sin comentarios: si necesitas pensar, no lo escribas.
- Si de verdad no hay nada que citar, deja el campo vacio.""

{{examples_block}}

RESPONDE SOLO con JSON:
{"fields": {"category": "ok|menor|grave", "subcategory": "ok|menor|grave", "urgency": "ok|menor|grave", "affected_elements": "ok|menor|grave", "risks": "ok|menor|grave", "location": "ok|menor|grave"}, "n_problems_texto": 0, "urgencias_detectadas": ["inmediata|urgente|alta|media|baja|planificable"], "expresion_citada": "frase literal del texto que fija la urgencia", "elementos_escenario": [{"elemento": "principal/subcategoria", "mencionado": "si|no|parcial"}], "peligros_texto": [{"peligro": "frase del texto", "riesgo": "eléctrico|estructural|sanitario|incendio|inundación|caída|accesibilidad|seguridad_persona"}], "riesgos_escenario": [{"riesgo": "eléctrico|estructural|sanitario|incendio|inundación|caída|accesibilidad|seguridad_persona|ninguno", "mencionado": "si|no|parcial", "cita": "frase literal del texto"}], "riesgos_extra": ["riesgo concreto nombrado y no presente en el escenario"], "inventado": {"elementos": [], "riesgos": [], "problemas": []}, "omitido": {"elementos": [], "riesgos": []}, "evidencia": "frase literal del texto, maximo 15 palabras, sin comillas dentro", "valid": true, "issues": []}
