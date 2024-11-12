import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import openai
import streamlit as st
import warnings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage

# Configura la tua logica di AI
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

vectorstore = Chroma(embedding_function=embeddings,persist_directory="./chroma_db_final_chunk")

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=openai_api_key)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_system_prompt = """
Comportati come un esperto in didattica.
Se la domanda è generica per esempio "consigliami qualche attività didattica da fare" oppure "cerco qualcosa" e frasi simili a queste e ti viene chiesto qualcosa senza specificare l'argomento da trattare, allora chiedimi l'argomento, l'età e la durata di quello che sto richiedendo.
Rileva la lingua della domanda e rispondi nella stessa lingua. 
Nella risposta includi sempre il link che hai a disposizione associato alla risorsa didattica e mettimi tutto in un elenco puntato nel seguenti campi: link, età, livello,furata, materiali.
Di seguito i dettagli per ogni link con i campi da mettere nell'elenco puntato:
'https://astroedu.iau.org/en/activities/2406/discover-earths-climate-with-a-balloon/':8-12,Middle School Primary,45 minuti,Image of climatic zones on Earth,Thread,Tape,Glue/sellotape,Ball/Balloon,Scissors,Marker /pencil,Two circular cut-outs (labelled ‘N’ and ‘S’),Labels (braille or printed).
'https://astroedu.iau.org/en/activities/2403/find-the-hidden-rainbows/':10-16, Middle School Secondary, 1 ora, Diffraction grating glasses (500 lines/mm), Various light sources (continuous emission like halogen or white LEDs, and line emission like fluorescent lamps), Power supply for light sources, Optional colored optical filters, Optional black background for spectrum observation.
'https://astroedu.iau.org/en/activities/2405/the-gravity-battle/':8-19, Informal   Middle School   Primary   Secondary, 1 ora, Wooden cubes, Nails, Copper wire, Aluminum foil, Water containers, Shoe boxes, Magnets, Glue, Sheets of paper, Inclined plane (half painted).
'https://astroedu.iau.org/en/activities/2404/skao-and-the-mysteries-of-invisible-light/':12-16, Informal   Middle School   Secondary, multiple days, Infrared camera, Black plastic bag, Radio Frequency (RF) detector, Spectrum analyser (optional), Parabolic microphone, Laser pointer, Optical fibre, Sound system with two input signals, Two mp3 players.
'https://astroedu.iau.org/en/activities/2402/chasing-the-moon/':14-19, Middle School   Secondary, multiple days, Cross-staff, Star charts, Protractor or compass, Paper, Pen or pencil.
'https://astroedu.iau.org/en/activities/2401/the-sky-at-your-fingertips/':10-14, Middle School  Primary, 2 hours, Stiff paper (A4, at least 200 g/m²), 40 cm ruler, Pen, Scalpel, Scissors, Adhesive tape, Cardboard, Cross-staff mask, Ruler mask.
'https://astroedu.iau.org/en/activities/2312/light-play/':6-14, Primary Middle School  Primary, multiple days, Cardboard boxes (approx. 40x40x60 cm), Translucent paper (e.g., tracing paper), Adjustable clamp lamps, Assorted materials for shadow creation (e.g., graters, strainers, nets), Colored translucent materials (e.g., colored plastic bottles), Reflective materials (e.g., mirrors, aluminum foil), Hot glue gun and glue sticks, Wire cutters, Scissors, Utility knives, Tape, DC electric motors with gear reduction (4-6 RPM), Battery holders and suitable batteries, Alligator clips.
'https://astroedu.iau.org/en/activities/2304/can-you-find-the-exoplanet/':14-19, Secondary, 3 hours, SalsaJ software, Computer with internet access, Spreadsheet software, Series of astronomical images (provided in the activity), Calculator, Pen and paper.
'https://astroedu.iau.org/en/activities/2307/how-do-telescopes-work/':10-12, Primary, 1 hour, 10-15 magnifying glasses, 10-15 small mirrors, 3-4 laser pointers, Glass of water, Spoon, Cups of soap and/or juice.
'https://astroedu.iau.org/en/activities/2306/orion-constellation-in-3d/':8-12, Primary, 2 hours, Styrofoam board (slightly larger than A4 size, 1-3 cm thick), 10 wooden skewers (20-30 cm long), Modeling clay (white, blue, light blue, orange, red), Ruler, Meter, Scissors or cutter, Tape or glue.
'https://astroedu.iau.org/en/activities/2308/the-sun-in-our-box/':10-14, Middle School, 3 hours, 3-4 sheets of colored cardboard (45 x 65 cm), Aluminum foil or cardboard, Wax paper, Ruler, Meter stick, Pin or needle, Pen, Tape, Scissors, Utility knife (for older students), A4 paper.
'https://astroedu.iau.org/en/activities/2301/become-a-geo-detective/':12-16, Middle School Secondary, 2 hours, Introduction video, Presentation slides (Introduction.ppt), Handout without solutions (Handout-without-solutions.pdf), Handout with solutions (Handout-with-solutions.pdf), Color markers.
'https://astroedu.iau.org/en/activities/2302/light-in-a-matchbox/':8-14, Primary Middle School, 1 hour 30 minutes, Matchbox (or similar small box), Black marker or black paint with brush, Cutting blade, Glue, Paper and colored pencils, CD or DVD.
'https://astroedu.iau.org/en/activities/2305/let-there-be-light-but-not-too-much/':6-12, Primary, 2 hours, Wooden board for the ground, Wooden board for the night sky, Blue paint, LED lights (e.g., Christmas tree lights), Black sandpaper, Reflective paper, Various lighting designs (as shown in the activity's images), Optional: printed card game for evaluation.
'https://astroedu.iau.org/en/activities/2303/moving-constellations/':10+, Middle School Secondary, 3 hours, Computer, Stellarium software, Presentation software (e.g., PowerPoint, Keynote), Introduction presentation about Ursa Major, Celestron SkyPortal app (optional).
'https://astroedu.iau.org/en/activities/age-that-crater/':4-14, Primary Middle School, 1 hour, Printed images of lunar craters, Ruler, Protractor, Calculator, Pen or pencil, Worksheet for recording observations.
'https://astroedu.iau.org/en/activities/2203/lets-play-with-powers-of-10/':12+, Middle School Secondary University, 2 hours, Printed card sets (Red: Sub-atomic to Molecular, Blue: Molecular to Human, Green: Human to Astronomical, Yellow: Astronomical to Cosmological), Post-it notes, Pens or pencils, Optional: internet access for research.
'https://astroedu.iau.org/en/activities/2002/misurare-la-velocita-media-di-una-cometa/':12+, Middle School Secondary, 1 hour 30 minutes, Personal computer with SalsaJ software installed, Spreadsheet software or graph paper, Printed images of comet C/2019 Y4 (optional), Printer (if using printed images), Calculator, Pen or pencil, Ruler.
'https://astroedu.iau.org/en/activities/2202/make-your-own-sun/':6-12, Primary, 1 hour 30 minutes, Paper to protect the workspace, Various colors of plasticine to represent different layers of the Sun, Scalpel or sharp-edged ruler for cutting the model in half (use with caution and adult supervision).
'https://astroedu.iau.org/en/activities/2201/hunting-for-spectra/':8-19, Middle School Secondary, 1 hour, Scissors, Ruler, Pen, Tape, Glue, A4 black cardboard, CD or DVD (clean), Printout of the paper spectroscope template (printed at 1:1 scale), Cutter or utility knife (to be used by an adult for safety reasons).
'https://astroedu.iau.org/en/activities/2001/driving-on-mars/':14-19, Middle School Secondary, 2 hours, Cardboard box (50 cm x 50 cm x 30 cm or larger), Cutters and scissors, Glue, Markers, Paint, Adhesive tape, String, ID cards of space agencies, Plates with the logos of one or more space agencies, "Water/rock" memory cards, Message cards of one or more space agencies, Obstacles (traffic cones, boxes).
'https://astroedu.iau.org/en/activities/2103/is-the-sun-rotating-follow-the-sunspots/':12-19, Middle School Secondary, 1 hour, Printed drawings of sunspots by Galileo Galilei, Printed set of images of the Sun by the Solar Dynamics Observatory (SDO), Computer with internet access (optional), SalsaJ software (optional), Telescope or solarscope for solar observation (optional).
'https://astroedu.iau.org/en/activities/2102/reading-the-rainbow/':14+, nformal   Middle School   Secondary   University, 1 hour 30 minutes, Single-axis diffraction gratings (one per student or shared), Incandescent bulb with clear glass, Dimmer (rheostat), Fluorescent light, Helium discharge tube/spectrum tube, At least one other discharge/spectrum tube, Colored pencils, Computer with internet access and projector.
'https://astroedu.iau.org/en/activities/2101/one-million-earths-inside-our-sun/':8-16, Primary Middle School Secondary, 30 minutes, 2 clear plastic fillable balls (30 cm in diameter), 14 liters of polystyrene beads (bean bag filler, around 3 mm in diameter), Dark blue acrylic paint, Cup, Bowl, Box with a lid, Hairdryer, Laundry net, Power drill, Rubber gloves, Sellotape, Funnel, Wallpaper paste, Yellow tissue paper, Tinfoil, Tape measure.
'https://astroedu.iau.org/en/activities/1801/measure-the-suns-rotation-period/':16-19, Secondary, 1 hour 30 minutes, Printed drawings of sunspots by Galileo Galilei, Printed images of the Sun by the Solar Dynamics Observatory (SDO), Plastic ruler, Computer with internet access (optional), SalsaJ software (optional), On-screen ruler (optional).
'https://astroedu.iau.org/en/activities/1747/dark-matter-and-dark-energy-part-1-discovering-the-main-components-of-the-universe/':12+, Middle School Secondary, 45 minutes, Computer and projector, Large round washbowl (minimum 30 cm diameter), Stretchy sheet (cut from a fitted bed sheet), Elastic band, Marbles of different sizes, Weighing scales, Intact and punctured balloons, Clear plastic string.
'https://astroedu.iau.org/en/activities/1749/birth-of-a-black-hole/':12-14, Middle School, 45 minutes, Round washing bowl (minimum 30 cm diameter), Stretchy sheet (cut from a fitted bed sheet), Elastic band, Marbles of varying weights (heavy, medium, light), Aluminum foil, Weighing scale, Small rectangular piece of cardboard, Projector, Student worksheet.
'https://astroedu.iau.org/en/activities/1751/hunting-for-black-holes-lower-secondary-level/':12-14, Middle School, 45 minutes, Heavy and light marbles, Large round washbowl (minimum 30 cm diameter), Stretchy sheet (cut from a fitted bed sheet), Elastic band, Wooden skewers, Cardboard, Orange highlighter, Chalk, Magnets, Ball bearing, Wine glass, Candle with holder (birthday cake type), Plastic cup, Lighter, Saucer, Colored tape, Projector, PowerPoint presentation, Student worksheet.
'https://astroedu.iau.org/en/activities/1748/dark-matter-dark-energy-part-2-understanding-the-nature-of-dark-matter-and-dark-energy/':12+, Middle School Secondary, 45 minutes, Computer and projector, Large round washbowl (minimum 30 cm diameter), Stretchy sheet (cut from a fitted bed sheet), Elastic band, Marbles of different sizes, Water, Small beads, Watercolor, Wine glass, Candle with holder (birthday cake type), Disposable cup, Lighter, Colored tape, Saucer, Magnet, Cardboard (with a nail taped on the back).
'https://astroedu.iau.org/en/activities/1648/navigate-like-a-viking-use-the-sun-not-your-phone/':12-16, Middle School Secondary, 1 hour 30 minutes, Worksheet, Cardboard (minimum 4 cm in diameter), Toothpick, Earth globe with stable mounting (e.g., inflatable), Compass (drawing tool), Lamp or spotlight, Scissors, Cutter or sharp knife, Glue, Blu Tack or similar adhesive.
'https://astroedu.iau.org/en/activities/1624/the-engine-of-life/':12-16, Middle School Secondary, 1 hour, Strong lamp or floodlight, Dimmer switch, Folding rule or yardstick, Photovoltaic cell with attached electric motor or fan, Pencils (regular and colored), Compass (drawing tool), Millimeter paper, Ruler, Calculator, Computer with internet access.
'https://astroedu.iau.org/en/activities/1636/the-big-meltdown/':8-16, Middle School Secondary, 45 minutes, Large transparent bowl, Ice, Large stone or similar item serving as a platform (about half the size of the bowl), Water, Overhead marker, Strong lamp.
'https://astroedu.iau.org/en/activities/1634/transforming-water-into-acid-and-back/':10-19, Middle School Secondary, 45 minutes, Distilled or demineralised water, Transparent cup or glass, Straw, Universal pH indicator (e.g., McCrumb) with corresponding pH scale, Alternative: pH indicator made from red cabbage, Small heater or stove (for the second part of the activity). 
'https://astroedu.iau.org/en/activities/1630/oceans-as-a-heat-reservoir/':12-19, Middle School Secondary, 1 hour 30 minutes, Strong lamp, Water, Dirt, soil, or sand, 2 bowls or trays (e.g., petri dishes), Stopwatch, Pen and paper, Colored pencils, Ruler, Thermometer, Calculator, Support to keep the thermometer upright (e.g., pin, paper clip). 
'https://astroedu.iau.org/en/activities/1628/where-on-earth-am-i/':14-19,Middle School Secondary, 1 hour 30 minutes, Worksheets, Compasses (drawing tool), Pencil, Ruler (at least 20 cm), Calculator. 
'https://astroedu.iau.org/en/activities/1621/valleys-deep-and-mountains-high/':14-19,Middle School Secondary, 2 hours, Worksheet, Landscape models (one per group of 2-4 students), Boxes (one per model), Wooden skewers (one per model), Rulers, Tape measures, Colored pencils, Calculators, Graph paper, Millimeter paper, Plotting paper, Computers with Microsoft Excel (version 2010 or later) for Activity 3. 
'https://astroedu.iau.org/en/activities/1618/a-view-from-above/':14-19,Middle School Secondary, 1 hour 30 minutes, Computer with LEO Works 4.0 software installed, Landsat satellite data files (Landsat Jan2002 and Landsat Jul2002), Student worksheet.
'https://astroedu.iau.org/en/activities/1620/the-climate-in-numbers-and-graphs/':14-19,Middle School Secondary, 2 hours, Worksheet, Graph paper, Pencils (black, blue, red, yellow), Ruler, Computers with spreadsheet software (e.g., MS Excel), Internet access, Pocket calculators.
'https://astroedu.iau.org/en/activities/1647/navigating-with-the-kamal-northern-hemisphere/':14-19,Middle School Secondary, 1 hour 30 minutes, Kamal or materials to build one (plywood or stiff cardboard, 50 cm cord, pencil, ruler, saw or scissors, drill or thick needle), Pencil, Torch (for outdoor activity). 
'https://astroedu.iau.org/en/activities/1622/big-telescopes-gravity/':12-16,Middle School Secondary, 1 hour 30 minutes, Computer/laptop with data logging software, Pair of connected light-gates, Clamp stand, Regular-shaped object to drop, Stopwatch, Newton meter, Objects of various masses, Hula hoop, Approximately 1m² of stretchy lycra material, Bulldog clips, Balls of different sizes and masses.
'https://astroedu.iau.org/en/activities/1646/the-quest-for-longitude/':14-19,Middle School Secondary, 2 hours, Printed worksheets, Crafting template and instructions for the longitude clock, Crafting knife, Scissors, Glue, Calculator, Pencil, Computer with Java runtime environment installed (for longitude clock app), Internet-enabled device (for online map service).
'https://astroedu.iau.org/en/activities/1619/the-intertropical-convergence-zone/':12-19,Middle School   Secondary, 1 hour, Paper handkerchief, napkin, or dual-chamber tea bag; Matches or lighter; China plate or tinfoil; Strong lamp (minimum 100 W); Scissors; Flat nose pliers (if available); Glue (for cardboard); Pencil or similar pointed object; Aluminum wrap of a tea light; Drawing pin; Cardboard tube (inner part of a kitchen roll); Black paint and brush or black colored paper; One piece of cardboard (approx. 1 cm wide, 8 cm long).
'https://astroedu.iau.org/en/activities/1643/country-movers-visualizing-spatial-scales-in-planetary-and-earth-sciences/':12+, Middle School Secondary, 2 hours, Computer/laptop/smartphone with internet access, Access to the Country Movers website (http://countrymovers.elte.hu), Projector (optional for class-wide activity), Paper, Pen.
'https://astroedu.iau.org/en/activities/1638/meteoroids-meteors-and-meteorites/':6-14, Primary Middle School, 1 hour 30 minutes, Rocks (preferably flint), Safety glasses, Cup or similar-sized freezer-proof container, Water, Gravel-sized rocks and/or sand, Freezer, Printed worksheets, Projector for watching videos and images.
'https://astroedu.iau.org/en/activities/1641/impact-craters/':10-16, Middle School Secondary, 1 hour, Plastic tray or bucket (at least 30 cm wide and 10 cm deep), Sand (enough to fill the container to a depth of at least 10 cm), Three rocks per group (weighing approximately 20g, 100g, and 200g), One rock weighing approximately 80g, Weighing scales, Ruler, Graph paper, Meter-long ruler, 30 cm ruler, Stand for holding the rulers (or tape to attach them vertically to the container), Projector/screen.
'https://astroedu.iau.org/en/activities/1642/creating-asteroids/':4-10, Primary, 1 hour 30 minutes, Images of asteroids (e.g., Asteroid Ida and the Antiope Doublet), Clay (a handful per participant), Paint brushes, Paint, Table lining/protector.
'https://astroedu.iau.org/en/activities/1644/childrens-planetary-maps-pluto-charon/':6-14, Primary Middle School, 2 hours, Map of Pluto and Charon for children (printed in large format or projected), Printed handouts, Printed worksheets. 
'https://astroedu.iau.org/en/activities/1718/childrens-planetary-maps-titan/':6-12,Middle School Primary, 2 hours, Map of Titan (printed in large format or projected), Printed handouts, Printed worksheets. 
'https://astroedu.iau.org/en/activities/1719/childrens-planetary-maps-io/':6-14, Primary Middle School, 2 hours, Map of Io (printed in large format or projected), Printed handouts, Printed worksheets.
'https://astroedu.iau.org/en/activities/1720/childrens-planetary-maps-the-moon/':6-14, Primary Middle School, 2 hours, Map of the Moon (printed in large format or projected), Printed handouts, Printed worksheets.
'https://astroedu.iau.org/en/activities/1721/childrens-planetary-maps-mars/':6-14, Primary Middle School, 2 hours, Map of Mars (printed in large format or projected), Mars Worksheet, Mars Handout.
'https://astroedu.iau.org/en/activities/1722/childrens-planetary-maps-venus/':6-14, Primary Middle School, 2 hours, Map of Venus (printed in large format or projected), Venus Handout, Venus Worksheet.
'https://astroedu.iau.org/en/activities/1645/navigation-in-the-ancient-mediterranean-and-beyond/':14-19,Middle School Secondary, 1 hour 30 minutes, Worksheet, Pair of compasses, Pencil, Ruler, Calculator, Protractor, Computer with MS Excel installed, Excel spreadsheet (provided in the activity's attachments), Portable red lamp (e.g., dimmed torch or torch with red filter), Magnetic compass (if available).
'https://astroedu.iau.org/en/activities/1703/the-4-point-backyard-diurnal-parallax-method/':16+, Secondary University, multiple days, Telescope (amateur-level, 8” or larger, polar-aligned, good clock-drive), CCD imager (preferably monochrome), PC (notebook), Planetarium software (e.g., C2A, SkyChart, Home Planet), CCD control software, Image-processing software (e.g., IRIS), Astrometric analysis software (e.g., X Parallax Viu, Astrometry.net), Spreadsheet, Time synchronization for CCD images (any Internet time service provider will do).
'https://astroedu.iau.org/en/activities/1616/evening-sky-watching-for-students/':4-8, Pre-school Primary, 30 minutes, Good weather conditions, Classroom with a large window or access to a garden/playground with a clear view of the sky, Ability to switch off bright outdoor lights (if possible).
'https://astroedu.iau.org/en/activity/street-lights-as-standard-candles/':14+, Secondary, 3 hours, Digital camera (capable of capturing raw format images), Computer with internet access, Image processing software (e.g., ImageJ or AstroImageJ), Spreadsheet software (e.g., Microsoft Excel, OpenOffice Calc, or Google Sheets).
'https://astroedu.iau.org/en/activities/1609/how-to-travel-on-earth-without-getting-lost/':8-14, Primary Middle School, 1 hour, Globe with marked lines of longitude and latitude, Lamp, Protractor, Three pencils, Sticker, Internet access to check precise latitude and longitude coordinates (e.g., http://www.findlatitudeandlongitude.com/), Night sky app for tablet or mobile device (e.g., Sky Map or Pocket Universe). Optional: String to wrap around the globe, Compass to show north, Quadrant.
'https://astroedu.iau.org/en/activities/1604/seasons-around-the-world/':6-12, Primary Middle School, 45 minutes, 3 cocktail sticks (or stickers for marking locations on a ball), Pen for drawing on an orange, Torch, Orange (or a ball as an alternative).
'https://astroedu.iau.org/en/activities/1603/investigating-the-atmosphere-air-takes-up-space/':6-10, Primary, 1 hour, Balloons, Plastic syringes (without needles) of two different sizes, Narrow straws, Plastic pipettes, Blu Tack, Bowl of water, Plastic or glass cup, Tissues.
'https://astroedu.iau.org/en/activities/1602/continental-climate-and-oceanic-climate/':6-10, Primary, 45 minutes, 2 clear plastic cups, 2 thermometers, Water, Soil, Sunlight or a bright lamp, Red, yellow, green, and blue coloring pencils, Worksheet. 
'https://astroedu.iau.org/en/activities/1617/create-your-own-astro-music/':8-19, Primary Middle School Secondary, 1 hour, One or more astronomical images, Method to display images (e.g., computer, projector, screen, or printouts), Musical instruments (e.g., drums, violins, flutes, trumpets, piano, tambourines, cymbals, shakers).
'https://astroedu.iau.org/en/activities/1418/star-hats/':4-6, Pre-school Primary, 30 minutes, Scissors, Coloured paper, Thick coloured paper (cardboard), Markers, Glue, Stapler, Optional: glitter glue, stickers, paint, paintbrushes.
'https://astroedu.iau.org/en/activities/1413/build-your-own-artificial-satellite/':8-10, Primary, 1 hour 30 minutes, Small carton, Toilet roll, Aluminium foil, Coloured paper, Iron wire, Skewers, Glue, Tape, Scissors, Plastic cups, Buttons or other decorative materials. 
'https://astroedu.iau.org/en/activities/1615/know-your-planets/':4-10, Primary, 1 hour, Card game PDF (one set per group), Planets and Sun model (e.g., Universe in a Box or other materials), Coloured pencils (one set per group), Scissors (one per student), Photos of the Solar System PDF (one set per group).
'https://astroedu.iau.org/en/activities/1614/sun-earth-and-moon-model/':8-10, Primary, 1 hour 30 minutes, Worksheet in PDF, A4 paper, Wooden skewer, Cocktail stick, Colouring pencils, Felt-tip pens or wax crayons, Scissors, String to attach cutouts to mobile, Sticky tape, Embroidery needles or paper punches (to share with the class), Stiff card to print memory game onto (per pair of students), One sheet of A2 paper (per class).
'https://astroedu.iau.org/en/activities/1613/make-a-star-lantern/':6-10, Primary, 1 hour 30 minutes, Worksheet (PDF), Coloured A3 card sheet (one per student), Lantern lights (one per student), Embroidery needles, Newspaper, Sticks to hold the lantern (one per student), Crepe paper and tissue paper in different colours, Brushes, A4 paper or printed sky maps, Luminous paint (or normal paint), Glue, Stapler, String, Camera (optional).
'https://astroedu.iau.org/en/activities/1612/history-of-the-universe/':8-14, Primary Middle School, 1 hour 30 minutes, Craft paper, Calculator or squared paper, Scissors, Glue, Colouring pencils or other colouring mediums (e.g., watercolours, pens, paint), 12 A4 sheets of plain or coloured paper, 1 coloured A4 sheet for the clock.
'https://astroedu.iau.org/en/activities/1611/living-in-the-milky-way/':6-10, Primary, 1 hour 30 minutes, Worksheet in PDF, Black A4 card sheet, Drawing compass, Scissors, White and red paint, Paintbrushes, Cotton wool, Glue, Red clay, 5 cocktail sticks, Fishing line. 
'https://astroedu.iau.org/en/activities/1610/lets-map-the-earth/':8-14, Primary Middle School, 1 hour, Aerial photograph of the city, Tracing paper, Ruler, Optional: calculator, 10 blank stickers or labels, World map, Pencils.
'https://astroedu.iau.org/en/activities/1608/making-a-sundial/':6-14, Primary Middle School, 1 hour, 12 large stones, Scissors, Glue, 150-centimetre long stick, Large protractor, Marker pen, Compass, Printed Sundial Worksheet PDF per student.
'https://astroedu.iau.org/en/activities/1607/what-is-a-constellation/',???????????
'https://astroedu.iau.org/en/activities/1606/what-is-time/':6-12, Primary Middle School, 1 hour, Two small light bottles of the same size, Stopwatch, A piece of cardboard slightly larger than the mouth of the bottle, Scissors, Sticky tape, Sand (enough to nearly fill a bottle), Worksheet printout, Pencil.
'https://astroedu.iau.org/en/activities/day-and-night-in-the-world/':6-10, Primary, 1 hour, Photographs of diurnal and nocturnal animals, Worksheet, 24 split pins, 2 cocktail sticks, Torch, Orange, Scissors, A4 paper, Thick paper, Colouring pencils, Embroidery needles.
'https://astroedu.iau.org/en/activities/1601/fizzy-balloons-co2-in-school/':8-14, Middle School Primary, 2 hours, Balloons, Funnel, Effervescent tablets (e.g., vitamin C tablets) or baking powder, Transparent 500 ml bottle, Water, Beakers, Tea light candle, Matches, Pair of tongs (or a wooden skewer), Lime water (prepared by mixing a spoonful of cement or mortar with about 250 ml water, letting the suspension settle, then filtering it using two paper coffee filters), Thick drinking straw.
'https://astroedu.iau.org/en/activities/lunar-day/':4-8, Primary, 30 minutes, Two paper plates (10 inches – 25.4 cm), A4 printouts of the Moon and the Earth, Scissors, Glue, Elastic bands, Access to Internet.
'https://astroedu.iau.org/en/activities/1501/how-many-stars-can-you-see-at-night/':12-14, Middle School, 45 minutes, Worksheet (in PDF or DOC format), Foldable Mini-Spectrometer (in PDF format), Black cardboard, CD, Scissors, Smartphone, Glue, Black adhesive tape, Access to Google Drive, Stellarium software, Computer, Data projector. 
'https://astroedu.iau.org/en/activities/1512/solar-system-model-on-a-city-map/':12-16,Middle School Secondary, 2 hours, City map, Pencil, Ball (~200 mm), Pinhead (~0.8 mm), Peppercorns (~2 mm), Poppy seed (~1 mm), Chestnut (~22 mm), Hazelnut (~18 mm), Peanuts (~8 mm), Sand grain (~0.6 mm), Optional: aluminum foil or play dough, Metric ruler, Calculator, Student Worksheet, Compass (optional).
'https://astroedu.iau.org/en/activities/1503/suns-shadow/':6-16, Primary Middle School Secondary, 1 day, Opaque objects (e.g., building, tree, person, stick), Paper (standard size, A1 or A0), Marker, Measuring tape, Rulers, Earth globe (or ball to represent the Earth), Flashlight. 
'https://astroedu.iau.org/en/activities/1505/solar-system-model/':4-10, Primary, 30 minutes, Plastic balls of various sizes, Paint and brushes, Images of planets, Clay or papier-mâché, Cotton. 
'https://astroedu.iau.org/en/activities/1502/lets-break-the-particles/':8-14, Primary Middle School, 1 hour 30 minutes, Plastic tubes (used by electricians), Glass marbles, Steel marbles, Adhesive putty (e.g., Patafix, Blu Tac), Hot glue, Magnets. 
'https://astroedu.iau.org/en/activities/1414/astropoetry-writing/':6-14, Primary Middle School, 1 hour 30 minutes, Pen or pencil, Notepad, Sky chart (printable or smartphone app), Stellarium software (optional). 
'https://astroedu.iau.org/en/activities/1412/blue-marble-in-empty-space/',????
'https://astroedu.iau.org/en/activities/1411/the-fibre-optic-cable-class/':12-19,Middle School Secondary, 1 hour, Ball or container (capable of containing a small note or item), Note from the teacher (or other fun thing to put in the ball, i.e. a small toy), Multiple mirrors (one for all students), Laser Pointer, Clear, plastic cord (that internally reflect laser light so it comes out the other end) or a length of fibre optic cable, Fibre optic lamp, Laser Safety Glasses (recommended), Laser Radio Communicator (optional).
'https://astroedu.iau.org/en/activities/1406/meet-our-home-planet-earth/':6-12, Primary, 1 hour 30 minutes, Thin wire (for hurricane feature), Cotton (for clouds and hurricane), Aluminium foil (for polar caps), Thick fabric (for continents), Plastic (for liquid water), Glue, Scissors, Pen, Earth Features PDF (print out attachment), Earth Mold PDF (print out two copies of attachment). 
'https://astroedu.iau.org/en/activities/1404/deadly-moons/':6-12, Primary, 1 hour, Black paper, White paper, Soft pastels, Crayons, Felt-tip pens, Paint brushes, Pencils, PowerPoint slideshow. 
'https://astroedu.iau.org/en/activities/1408/meet-our-neighbours-moon/':6-12, Primary, 2 hours, Grey felt (for Moon surface), Thick fabric (for mountains), Thin fabric (for mare features), Curved sequins (for craters), Glue, Pen, Scissors, Moon Features PDF, Moon Featureless PDF (two copies).
'https://astroedu.iau.org/en/activities/1409/build-a-safe-sun-viewer/':6-12,Middle School Primary, 30 minutes, Cardboard box, White cardboard, Scissors, Tape, Aluminum foil. 
'https://astroedu.iau.org/en/activities/1410/coma-cluster-of-galaxies/',???
'https://astroedu.iau.org/en/activities/1404/deadly-moons/':6-12, Primary, 1 hour, Black paper, White paper, Soft pastels, Crayons, Felt-tip pens, Paint brushes, Pencils, PowerPoint slideshow. 
'https://astroedu.iau.org/en/activities/1403/globe-at-night-activity-guide/',
'https://astroedu.iau.org/en/activities/1402/how-light-pollution-affects-the-stars-magnitude-readers/',
'https://astroedu.iau.org/en/activities/1401/snakes-ladders-game/',
'https://astroedu.iau.org/en/activities/1311/lunar-landscape/',
'http://astroedu.iau.org/en/activities/why-do-we-have-day-and-night/',
'https://astroedu.iau.org/en/activities/1308/meet-our-neighbours-sun/',
'https://astroedu.iau.org/en/activities/1307/glitter-your-milky-way/',
'https://astroedu.iau.org/en/activities/1306/star-in-a-box-advanced/',
'https://astroedu.iau.org/en/activities/1301/counting-sunspots/',
'https://astroedu.iau.org/en/activities/1305/measure-the-solar-diameter/',
'https://astroedu.iau.org/en/activities/1302/star-in-a-box-high-school/',
'https://astroedu.iau.org/en/activities/1304/model-of-a-black-hole/',
'https://astroedu.iau.org/en/activities/1303/design-your-alien/'.

Quando mi dai come risposta uno di questi link mettimi sempre un riassunto della risorsa didattica in questione dopo l'elenco puntato.

Questi quattro link 
'https://astroedu.iau.org/en/activities/meet-our-home-planet-earth/',
'https://astroedu.iau.org/en/activities/meet-our-neighbours-moon/',
'https://astroedu.iau.org/en/activities/meet-our-neighbours-sun/',
'https://astroedu.iau.org/en/activities/discover-earths-climate-with-a-balloon/'
che fanno parte di tutti i link che hai a disposizione si riferiscono all'astronomia tattile quindi se ti viene richiesta una 
"attività per ciechi", "attività per ipovedenti", "attività per non vedenti", "attività tattile" rispondimi con uno di questi link 
in base alla domanda che ti è stata fatta dove è stato specificato l'argomento. Se la domanda che ti viene fatto non riguarda nessun 
argomento di questi link, allora rispondimi che non hai attività per quell'argomento per ragazzi ciechi, ipovedenti, non vedenti, tattile.

Traduci la risposta nella stessa lingua della domanda.

Context: {context}
Answer:

"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Configura la pagina
#st.set_page_config(page_title="AstroEdu AI Assistant", layout="wide")

# Intestazione
#st.markdown("<h1 style='text-align: center; color: #0004ff;'>Welcome to AstroEDU AI Assistant!</h1>", unsafe_allow_html=True)
st.image("./LOGO2.webp", use_column_width=True) 

# Sezione di Benvenuto
st.markdown("<h2 style='color: #FFA500;'>Welcome to AstroEDU AI Assistant!</h2>", unsafe_allow_html=True)
st.markdown("I'm here to help you find and make the best use of educational materials from AstroEDU.<br>How can I assist you? If you want, speak to me in your language!", unsafe_allow_html=True)

# Funzione per ottenere la risposta dall'assistente AI
def get_ai_response(question, chat_history):
    response = rag_chain.invoke({
        "chat_history": chat_history,
        "input": question
    })
    return response['answer']

# Funzione per gestire l'invio dei messaggi tramite il campo di input della chat
def chat_actions():
    user_input = st.session_state["chat_input"]
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    
    chat_history = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state["chat_history"]]
    ai_response = get_ai_response(user_input, chat_history)
    
    st.session_state["chat_history"].append({"role": "assistant", "content": ai_response})


# Inizializza la cronologia della chat se non esiste
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Campo di input per i messaggi della chat
st.chat_input("Enter your message", on_submit=chat_actions, key="chat_input")

# Visualizza la cronologia dei messaggi della chat
for i in st.session_state["chat_history"]:
    with st.chat_message(name=i["role"]):
        st.write(i["content"])     
