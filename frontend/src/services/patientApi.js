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

export const getPatientListCacheKey = (page = 1, pageSize = 20, searchQuery = '', processingStatus = 'all') =>
    buildCacheKey('patients', { page, pageSize, searchQuery, processingStatus });

export const getCachedPatientList = (page = 1, pageSize = 20, searchQuery = '', processingStatus = 'all') =>
    getCachedValue(getPatientListCacheKey(page, pageSize, searchQuery, processingStatus));

export const getChatUserId = () => {
    const storageKey = 'medsight_chat_user_id';
    const existingId = localStorage.getItem(storageKey);
    if (existingId) return existingId;

    const newId = crypto.randomUUID();
    localStorage.setItem(storageKey, newId);
    return newId;
};

export const getPatients = async (page = 1, pageSize = 20, processingStatus = 'all') => {
    try {
        const params = new URLSearchParams({
            page: String(page),
            page_size: String(pageSize),
        });
        if (processingStatus !== 'all') {
            params.set('processing_status', processingStatus);
        }

        const data = await fetchJsonWithCache(
            getPatientListCacheKey(page, pageSize, '', processingStatus),
            `${API_BASE_URL}/api/v1/patients/list/infor?${params.toString()}`
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

export const searchPatients = async (searchQuery, page = 1, pageSize = 20, processingStatus = 'all') => {
    try {
        const normalizedSearchQuery = searchQuery.trim();
        const params = new URLSearchParams({
            page: String(page),
            page_size: String(pageSize),
            search: normalizedSearchQuery,
        });
        if (processingStatus !== 'all') {
            params.set('processing_status', processingStatus);
        }
        return await fetchJsonWithCache(
            getPatientListCacheKey(page, pageSize, normalizedSearchQuery, processingStatus),
            `${API_BASE_URL}/api/v1/patients/list/infor?${params.toString()}`
        );
    } catch (error) {
        console.error('Error searching patients:', error);
        throw error;
    }
};

export const createPatient = async (payload) => {
    const response = await fetch(`${API_BASE_URL}/api/v1/patients/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Create patient failed: ${response.status} ${errorText}`);
    }

    patientResponseCache.clear();
    return response.json();
};

export const createCase = async (payload) => {
    const response = await fetch(`${API_BASE_URL}/api/v1/cases/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Create case failed: ${response.status} ${errorText}`);
    }

    patientResponseCache.clear();
    return response.json();
};

export const importPatientCase = async ({ name, age, gender, phoneNumber, diagnosis, findings, file }) => {
    const formData = new FormData();
    formData.append('name', name);
    if (age !== '' && age !== null && age !== undefined) {
        formData.append('age', String(age));
    }
    if (gender) {
        formData.append('gender', gender);
    }
    if (phoneNumber) {
        formData.append('phone_number', phoneNumber);
    }
    if (diagnosis) {
        formData.append('diagnosis', diagnosis);
    }
    if (findings) {
        formData.append('findings', findings);
    }
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/api/v1/patients/import-case`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Import patient case failed: ${response.status} ${errorText}`);
    }

    patientResponseCache.clear();
    return response.json();
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

export const sendOpenAIChatMessage = async ({ sessionId, message, imageUrl, mode, patientContext, currentAnnotations, submittedDiagnosis }) => {
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
        throw new Error(`AI chat failed: ${response.status} ${errorText}`);
    }

    return response.json();
};

export const sendMedGemmaChatMessage = sendOpenAIChatMessage;

const parseSseBlock = (block) => {
    let event = 'message';
    const dataLines = [];

    block.split(/\r?\n/).forEach((line) => {
        if (!line || line.startsWith(':')) return;
        if (line.startsWith('event:')) {
            event = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
            dataLines.push(line.slice(5).trimStart());
        }
    });

    if (dataLines.length === 0) return null;

    const rawData = dataLines.join('\n');
    try {
        return { event, data: JSON.parse(rawData) };
    } catch (error) {
        throw new Error(`Invalid streaming event payload: ${rawData}`);
    }
};

export const streamOpenAIChatMessage = async ({
    sessionId,
    message,
    imageUrl,
    mode,
    patientContext,
    currentAnnotations,
    submittedDiagnosis,
    onUserMessage,
    onDelta,
    onDone,
    onError,
    signal
}) => {
    const response = await fetch(`${API_BASE_URL}/api/v1/education/sessions/${sessionId}/messages/stream`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream',
        },
        body: JSON.stringify({
            message,
            image_url: imageUrl,
            mode,
            patient_context: patientContext || null,
            current_annotations: currentAnnotations || [],
            submitted_diagnosis: submittedDiagnosis || null
        }),
        signal,
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`AI chat stream failed: ${response.status} ${errorText}`);
    }
    if (!response.body) {
        throw new Error('AI chat stream failed: empty response body');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    const handleEvent = ({ event, data }) => {
        if (event === 'user_message') {
            onUserMessage?.(data);
            return;
        }
        if (event === 'delta') {
            onDelta?.(data.delta || '');
            return;
        }
        if (event === 'done') {
            onDone?.(data);
            return;
        }
        if (event === 'error') {
            const error = new Error(data.detail || 'AI chat stream failed');
            onError?.(error);
            throw error;
        }
    };

    while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

        const blocks = buffer.split(/\r?\n\r?\n/);
        buffer = blocks.pop() || '';

        for (const block of blocks) {
            const parsed = parseSseBlock(block);
            if (parsed) handleEvent(parsed);
        }

        if (done) break;
    }

    if (buffer.trim()) {
        const parsed = parseSseBlock(buffer);
        if (parsed) handleEvent(parsed);
    }
};

export const generateMedicalReport = async ({ caseId, patientHistory = null, aiFindings = null }) => {
    const response = await fetch(`${API_BASE_URL}/api/v1/reports/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            case_id: caseId,
            patient_history: patientHistory,
            ai_findings: aiFindings,
        }),
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Report generation failed: ${response.status} ${errorText}`);
    }

    patientResponseCache.clear();
    return response.json();
};

export const analyzeCase = async (caseId, { forceRerun = false, confThres = 0.1, iouThres = 0.45 } = {}) => {
    const params = new URLSearchParams({
        force_rerun: forceRerun,
        conf_thres: confThres,
        iou_thres: iouThres,
    });

    const response = await fetch(
        `${API_BASE_URL}/api/v1/disease-detection/${caseId}?${params}`,
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

export const searchSimilarCases = async ({ caseId = null, imagePath = null, topK = 6 } = {}) => {
    const key = buildCacheKey('similarity-search', {
        caseId,
        imagePath,
        topK,
    });

    const cached = getCachedValue(key);
    if (cached) {
        return cached;
    }

    const existingRequest = inFlightRequests.get(key);
    if (existingRequest) {
        return existingRequest;
    }

    const request = fetch(`${API_BASE_URL}/api/v1/similarity/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            case_id: caseId,
            image_path: imagePath,
            top_k: topK,
        }),
    })
        .then(async (response) => {
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Similarity search failed: ${response.status} ${errorText}`);
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

export const generateSimilarityCam = async ({ caseId, similarCaseId }) => {
    const response = await fetch(
        `${API_BASE_URL}/api/v1/analysis/cam-inference/${caseId}/similar/${similarCaseId}`,
        { method: 'POST' }
    );

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Similarity CAM failed: ${response.status} ${errorText}`);
    }

    return response.json();
};
