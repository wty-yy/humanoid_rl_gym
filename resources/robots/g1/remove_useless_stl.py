from pathlib import Path

xmls = ""
files = [
    *Path(__file__).parent.glob("*.xml"),
    *Path(__file__).parent.glob("*.urdf")
]
for xml in files:
    print(xml)
    with open(xml, "r") as f:
        xmls += f.read()

meshes_path = Path(__file__).parent / "meshes"
for mesh in meshes_path.iterdir():
    if mesh.name not in xmls:
        print(f"Removing {mesh.name}...")
        mesh.unlink()
