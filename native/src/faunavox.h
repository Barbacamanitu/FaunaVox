/* C header for faunavox.dll (FaunaVox). See src/ffi.rs for details. MIT License, Copyright (c) 2026 Russ Martelli. */
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct CvCreature CvCreature;
typedef struct CvBuffer CvBuffer;

const char* cv_last_error(void);          /* "" if the last call on this thread succeeded */
const char* cv_version(void);
int32_t     cv_sample_rate(void);         /* 44100 */
const char* cv_intent_names(void);        /* comma-separated */
const char* cv_preset_names(void);        /* comma-separated */

CvCreature* cv_creature_from_json(const char* json);
CvCreature* cv_creature_from_preset(const char* name);
CvCreature* cv_creature_clone(const CvCreature* c);
void        cv_creature_free(CvCreature* c);
int32_t     cv_creature_set(CvCreature* c, const char* field, double value);            /* 0 ok, -1 error */
int32_t     cv_creature_set_json(CvCreature* c, const char* field, const char* json);   /* 0 ok, -1 error */
char*       cv_creature_to_json(const CvCreature* c);   /* free with cv_string_free */
void        cv_string_free(char* s);
char*       cv_english_to_phrase(const char* text); /* English -> speak phrase; free with cv_string_free */

/* From the voice section of a genome. body_json: {"mass", "tract_length" or "head_length" + "neck_length",
   "source", "shape", "lips", "teeth"}. versions_json may be NULL (built-in versions) or
   {"decoder": 2, "space": 1, "field": 1}; "field": false decodes without steering. NULL on error. */
CvCreature* cv_creature_from_genome(const char* body_json, const char* dna, const char* versions_json);
char*       cv_genome_report(const CvCreature* c);  /* decoder report JSON; free with cv_string_free */
int32_t     cv_field_version(void);
int32_t     cv_decoder_version(void);

/* Mono float samples at cv_sample_rate(), peak-normalized to 0.9. NULL on error.
   CPU work: call from a worker thread, never the audio thread. */
CvBuffer*    cv_render(const CvCreature* c, const char* intent, uint64_t seed, float distance, float reverb);
int32_t      cv_buffer_len(const CvBuffer* b);
const float* cv_buffer_data(const CvBuffer* b);
int32_t      cv_buffer_copy(const CvBuffer* b, float* dst, int32_t capacity);
void         cv_buffer_free(CvBuffer* b);
#ifdef __cplusplus
}
#endif
