#include "ccv.h"
#include <sys/time.h>
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

inline float IoU(ccv_rect_t a, ccv_rect_t b)
{
    int in_x = ccv_max(a.x, b.x);
    int in_y = ccv_max(a.y, b.y);
    int in_w = ccv_min(a.x + a.width, b.x + b.width) - in_x;
    if (in_w < 0)
        return 0.;
    int in_h = ccv_min(a.y + a.height, b.y + b.height) - in_y;
    if (in_h < 0)
        return 0.;
    float in_size = in_w * in_h; // float for conversion
    return in_size / (a.width * a.height + b.width * b.height - in_size);
}

ccv_dpm_param_t ccv_dpm_custom_params = {
	.interval = 8,
	.min_neighbors = 1,
	.flags = 0,
	.threshold = 0.8, // 0.8
};

static unsigned int get_current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int main(int argc, char** argv)
{
	assert(argc >= 3);
	int i, j;
    float iou_thre = 0.5;
	ccv_enable_default_cache();
	ccv_dense_matrix_t* image = 0;
	ccv_read(argv[1], &image, CCV_IO_ANY_FILE);
	ccv_dpm_mixture_model_t* model = ccv_dpm_read_mixture_model(argv[2]);
	if (argc >= 4)
		ccv_dpm_custom_params.threshold = atof(argv[3]);
    if (argc == 5)
    	iou_thre = atof(argv[4]); // iou threshold
	if (image != 0)
	{
		unsigned int elapsed_time = get_current_time();
		ccv_array_t* seq = ccv_dpm_detect_objects(image, &model, 1, ccv_dpm_custom_params);
		elapsed_time = get_current_time() - elapsed_time;
		if (seq)
		{
			for (i = 0; i < seq->rnum; i++)
			{
				ccv_root_comp_t* comp = (ccv_root_comp_t*)ccv_array_get(seq, i);
				printf("%d %d %d %d %f %d\n", comp->rect.x, comp->rect.y, comp->rect.width, comp->rect.height, comp->classification.confidence, comp->pnum);
				for (j = 0; j < comp->pnum; j++)
					printf("| %d %d %d %d %f\n", comp->part[j].rect.x, comp->part[j].rect.y, comp->part[j].rect.width, comp->part[j].rect.height, comp->part[j].classification.confidence);
			}
			printf("total : %d in time %dms\n", seq->rnum, elapsed_time);
			ccv_array_free(seq);
		} else {
			printf("elapsed time %dms\n", elapsed_time);
		}
		ccv_matrix_free(image);
	} else {
		FILE* r = fopen(argv[1], "rt");
        if(r)
		{
			size_t len = 1024;
			char* file = (char*)malloc(len);
			ssize_t read;
            int fa = 0, tp = 0, gt_num = 0;
			while((read = getline(&file, &len, r)) != -1)
			{
				while(read > 1 && isspace(file[read - 1]))
					read--;
				file[read] = 0;
                char* pch;
                pch = strtok(file, " ");
                image = 0;
				ccv_read(pch, &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);
				printf("img file: %s, ", pch);
                assert(image != 0);
                pch = strtok(NULL, " ");
                int obj_num = atoi(pch); // object number in the img
                printf("obj_num=%d, ", obj_num);
                gt_num += obj_num;
                ccv_rect_t* obj_pos = (ccv_rect_t*)malloc(obj_num * sizeof(ccv_rect_t));
                for (i = 0; i < obj_num; i += 4)
                {// get positions
                    pch = strtok(NULL, " ");
                    obj_pos[i].x = atoi(pch);
                    pch = strtok(NULL, " ");
                    obj_pos[i].y = atoi(pch);
                    pch = strtok(NULL, " ");
                    obj_pos[i].width = atoi(pch);
                    pch = strtok(NULL, " ");
                    obj_pos[i].height = atoi(pch);
                }
                ccv_array_t* seq = ccv_dpm_detect_objects(image, &model, 1, ccv_dpm_custom_params);
				printf("detected object number= %d\n", seq->rnum);
                if (seq != 0)
				{
					for (i = 0; i < seq->rnum; i++)
					{
						ccv_root_comp_t* comp = (ccv_root_comp_t*)ccv_array_get(seq, i);
					    float iou = 0;
                        for (j = 0; j < obj_num; j += 1)
                        {
                             iou = MAX(iou, IoU(obj_pos[j], comp->rect));
                        }
                        if (iou >= iou_thre)
                        {
                            tp += 1;
                        }
                        else {
                            fa += 1;
                        }
                        //printf("%s %d %d %d %d %f %d\n", file, comp->rect.x, comp->rect.y, comp->rect.width, comp->rect.height, comp->classification.confidence, comp->pnum);
						//for (j = 0; j < comp->pnum; j++)
							//printf("| %d %d %d %d %f\n", comp->part[j].rect.x, comp->part[j].rect.y, comp->part[j].rect.width, comp->part[j].rect.height, comp->part[j].classification.confidence);
					}
					ccv_array_free(seq);
				}
				ccv_matrix_free(image);
			}
			printf("precision: %.2f\%, recall: %.2f\%\n", tp * 100.0 / (tp + fa), tp * 100.0 / gt_num);
            free(file);
			fclose(r);
		}
	}
	ccv_drain_cache();
	ccv_dpm_mixture_model_free(model);
	return 0;
}
