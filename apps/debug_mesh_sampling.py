import trimesh
import os 


if __name__ == "__main__":
	base = "./training_data"
	subjects = os.listdir(os.path.join(base, "GEO", "OBJ"))
	for subject in subjects:
		mesh = trimesh.load(os.path.join(base, "GEO", "OBJ", subject, "%s_posed.obj" % subject), process=False, maintain_order=True, skip_uv=True)
		surface_points, surface_points_face_indices = trimesh.sample.sample_surface(mesh, 32000)
		surface_points_faces = mesh.faces[surface_points_face_indices]
        surface_points_vertices_indices = []
        
        for single_face in surface_points_faces:
            surface_points_vertices_indices.append(min(single_face))

        m1 = max(surface_points_vertices_indices)
        with open(os.path.join(base, "PART", subject, "%s_part.json" % subject.split('_')[0])) as f: 
    		json_data = json.load(f)
    	m2 = max(map(int, json_data.keys()))

    	if len(mesh.vertices) != len(json_data.keys()):
    		print(subject, "has unlabelled vertices")

    	if m1 > m2:
    		print(subject, "sampling error")


