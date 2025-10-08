# Instancias

Cada carpeta corresponde a una instancia a resolver con el modelo que propongan.
Cada instancia tiene los siguientes archivos .csv:

## overview

Archivo con datos generales del ruteo. Columnas:

* depot_latitude: latitud del depot (donde parten y terminan las rutas)
* depot_longitude: longitud del depot (donde parten y terminan las rutas)
* start_at: hora a la que empieza la jornada, vehículos no pueden salir del depot antes
* end_at: hora a la que termina la jornada, vehículos no pueden regresar al depot después
* demands: número de demandas a rutar
* vehicles: número de vehículos disponibles
* expected_matrix_size: tamaño de matriz esperada (1 depot + # demandas)^2, chequeo de sanidad
* distances_size: tamaño de matriz de distancias generada
* times_size: tamaño de matriz de tiempos generada

## demandas

Archivo con datos de las demandas a rutear. Columnas:

* id: id de la demanda
* latitude: latitud de la demanda (donde se debe entregar)
* longitude: longitud de la demanda (donde se debe entregar)
* size: tamaño en kilogramos
* stop_time: duración de la detención para entregar, en segundos
* tw_start: de existir, hora de inicio de la ventana de atención. No se puede entregar antes (se puede esperar)
* tw_end: de existir, hora de fin de la ventana de atención. No se puede entregar después.

## vehicles

Archivo con datos de la flota de vehículos disponible para rutear. Columnas:

* id: id del vehículo
* capacity: capacidad en kilogramos

## distances

Archivo con las distancias entre pares de puntos. Puntos son el depot + demandas. Columnas:

* origin_latitude
* origin_longitude
* destination_latitude
* destination_longitude
* distance: distancia entre origen y destino en metros

## times

Archivo con los tiempos entre pares de puntos. Puntos son el depot + demandas. Columnas:

* origin_latitude
* origin_longitude
* destination_latitude
* destination_longitude
* time: tiempo entre origen y destino en segundos

# Constantes

Transversal a todas las instancias.

## costs

Archivo con los costos fijos y variables. Columnas:

 * fixed_route_cost: costo fijo por cada ruta creada, en CLP
 * cost_per_meter: costo variable por distancia recorrida, en CLP/metros
 * cost_per_vehicle_capacity: costo variable por el tamaño de los vehículos usados, en CLP/kgs