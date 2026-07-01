/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright 2025-2026 NXP
 * Copyright (C) 2021, Raspberry Pi (Trading) Ltd.
 *
 * preview.cpp - preview window interface
 */
#include <map>

#include "preview.hpp"
#include "mesh.hpp"

Preview *make_null_preview(int camera_num, int detector_type);
Preview *make_egl_preview(int camera_num, int detector_type);
Preview *make_drm_preview(int camera_num, int detector_type);

Preview *make_preview(int camera_num, int preview_type, int detector_type)
{
	try
	{
		Preview *p = NULL;
		if(preview_type == 0){
			p = make_egl_preview(camera_num, detector_type);
		} else if(preview_type == 1) {
			p = make_drm_preview(camera_num, detector_type);
		} else {
			p = make_null_preview(camera_num, detector_type);
		}
		if (p)
			printf("Made DRM preview window");
		return p;
	}
	catch (std::exception const &e)
	{
		printf("Preview window unavailable");
		return make_null_preview(camera_num, detector_type);
	}

	return nullptr; // prevents compiler warning in debug builds
}
