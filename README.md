# public-resources
Resources to be published and made public (e.g. Fruit Swarm Data)


## Data Format
The data for these packages comes in a JSON format. Each key in the JSON array is a Question ID,
 which is a unique ID that describes a single swarm answering one question. 
 
 The value of each key is a dictionary containing the:
 1) Answer Choices, in a list format, listed clockwise from the bottom-left node of the hex. 
 2) Question Title, in a string format
 3) Magnet Data, an (M timesteps x N users) array that describes the angle (in degrees)
 that each user was pulling towards at each timestep. Entry m,n is the mth timestep (each timestep is 0.25 seconds)
 and the nth user. Angular measurement in the Swarm client starts out at what would be traditionally considered 
 180 degrees and rotates clockwise, so 45 degrees in the Swarm system corresponds to 135 degrees in the standard system. An angle of -1 indicates 
 that the user was not pulling at that timestep.  
 4) Puck Data, an (M timesteps x 2 columns) array that describes the position of the puck in (x,y) coordinates at each timestep. 
 The x coordinate is standard, but the y coordinate is reversed: 0 is at the top of the browser, and 100 is 100 pixels below the top of the browser. 
 5) Factions (Factions Data), an (M timesteps x N users) array that describes the index of the answer that each user was 
 pulling for at each timestep. NaNs in this array indicate that the user was either not pulling, or the user was pulling between answers, 
 so the Faction they were pulling for cannot be determined.  
  
 