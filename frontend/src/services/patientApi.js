const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const PATIENT_CACHE_TTL_MS = 5 * 60 * 1000;

const patientResponseCache = new Map();
const inFlightRequests = new Map();

const buildCacheKey = (type, params) => `${type}:${JSON.stringify(params)}`;

const getCachedValue = (key) => {
    const cached = patientResponseCache.get(key);
    if (!cached) return null;

    if (Date.now() - cached.timestamp > PATIENT_CACHE_TTL_MS) {
        patientResponseCache.delete(key);
        return null;
    }

    return cached.data;
};

const setCachedValue = (key, data) => {
    patientResponseCache.set(key, {
        data,
        timestamp: Date.now(),
    });
};

const fetchJsonWithCache = async (key, url) => {
    const cached = getCachedValue(key);
    if (cached) {
        return cached;
    }

    const existingRequest = inFlightRequests.get(key);
    if (existingRequest) {
        return existingRequest;
    }

    const request = fetch(url)
        .then(async (response) => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            setCachedValue(key, data);
            return data;
        })
        .finally(() => {
            inFlightRequests.delete(key);
        });

    inFlightRequests.set(key, request);
    return request;
};

export const getPatientListCacheKey = (page = 1, pageSize = 20, searchQuery = '') =>
    buildCacheKey('patients', { page, pageSize, searchQuery });

export const getCachedPatientList = (page = 1, pageSize = 20, searchQuery = '') =>
    getCachedValue(getPatientListCacheKey(page, pageSize, searchQuery));

export const getChatUserId = () => {
    const storageKey = 'medsight_chat_user_id';
    const existingId = localStorage.getItem(storageKey);
    if (existingId) return existingId;

    const newId = crypto.randomUUID();
    localStorage.setItem(storageKey, newId);
    return newId;
};

export const getPatients = async (page = 1, pageSize = 20) => {
    try {
        const data = await fetchJsonWithCache(
            getPatientListCacheKey(page, pageSize, ''),
            `${API_BASE_URL}/api/v1/patients/list/infor?page=${page}&page_size=${pageSize}`
        );

        console.log('API Response:', data);
        return data;
    } catch (error) {
        console.error('Error fetching patients:', error);
        throw error;
    }
};

export const getPatientDetail = async (patientId) => {
    try {
        return await fetchJsonWithCache(
            buildCacheKey('patient-detail', { patientId }),
            `${API_BASE_URL}/api/v1/patients/${patientId}/infor`
        );
    } catch (error) {
        console.error('Error fetching patient detail:', error);
        throw error;
    }
};

export const getProxiedImageUrl = (s3Url) => {
    if (!s3Url) return null;
    return s3Url;
};

export const getDicomImageUrl = (dicomUrl) => {
    return getProxiedImageUrl(dicomUrl);
};

export const searchPatients = async (searchQuery, page = 1, pageSize = 20) => {
    try {
        const normalizedSearchQuery = searchQuery.trim();
        return await fetchJsonWithCache(
            getPatientListCacheKey(page, pageSize, normalizedSearchQuery),
            `${API_BASE_URL}/api/v1/patients/list/infor?page=${page}&page_size=${pageSize}&search=${encodeURIComponent(normalizedSearchQuery)}`
        );
    } catch (error) {
        console.error('Error searching patients:', error);
        throw error;
    }
};

export const resolveChatSession = async ({ caseId, sessionType = 'tutoring' }) => {
    const response = await fetch(`${API_BASE_URL}/api/v1/education/sessions/resolve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            user_id: getChatUserId(),
            case_id: caseId || null,
            session_type: sessionType
        })
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to resolve chat session: ${response.status} ${errorText}`);
    }

    return response.json();
};

export const sendMedGemmaChatMessage = async ({ sessionId, message, imageUrl, mode, patientContext, currentAnnotations, submittedDiagnosis }) => {
    const response = await fetch(`${API_BASE_URL}/api/v1/education/sessions/${sessionId}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            message,
            image_url: imageUrl,
            mode,
            patient_context: patientContext || null,
            current_annotations: currentAnnotations || [],
            submitted_diagnosis: submittedDiagnosis || null
        })
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`MedGemma chat failed: ${response.status} ${errorText}`);
    }

    return response.json();
};

export const analyzeCase = async (caseId, { forceRerun = false, confThres = 0.25, iouThres = 0.45 } = {}) => {
    const params = new URLSearchParams({
        force_rerun: forceRerun,
        conf_thres: confThres,
        iou_thres: iouThres,
    });

    const response = await fetch(
        `${API_BASE_URL}/api/v1/analysis/localize/${caseId}?${params}`,
        { method: 'POST' }
    );

    if (!response.ok) {
        const errorText = await response.text();
        let detail = errorText;
        try {
            const parsed = JSON.parse(errorText);
            detail = parsed.detail || errorText;
        } catch (_) {
            // Ignore JSON parse errors and keep the raw response text.
        }
        throw new Error(`Analyze failed: ${response.status} - ${detail}`);
    }

    return response.json();
};
