/**
 * Mock API cho PACS và VNA
 * Mô phỏng việc test connection và lưu cấu hình
 */

/**
 * Mock PACS Test Connection
 * @param {Object} config - Cấu hình PACS
 * @param {string} config.host - PACS host
 * @param {number} config.port - PACS port
 * @param {string} config.localAETitle - Local AE Title
 * @param {string} config.remoteAETitle - Remote AE Title
 * @returns {Promise} - Promise với kết quả test
 */
export const mockPacsTest = (config) => {
  return new Promise((resolve, reject) => {
    // Mô phỏng thời gian request: 1.5-2.5 giây
    const delay = 1500 + Math.random() * 1000;

    setTimeout(() => {
      // Validation đơn giản
      if (!config.host || !config.port) {
        reject({
          success: false,
          message: "Host và Port không được để trống",
        });
        return;
      }

      // Rule: host phải chứa dấu "." để giả lập domain/IP hợp lệ
      if (!config.host.includes(".")) {
        reject({
          success: false,
          message:
            "Host không hợp lệ. Vui lòng nhập địa chỉ IP hoặc domain (VD: 192.168.1.100)",
        });
        return;
      }

      // Rule: port phải từ 1-65535
      if (config.port < 1 || config.port > 65535) {
        reject({
          success: false,
          message: "Port phải trong khoảng 1-65535",
        });
        return;
      }

      // Rule: AE Title không được rỗng
      if (!config.localAETitle || !config.remoteAETitle) {
        reject({
          success: false,
          message: "Local AE Title và Remote AE Title không được để trống",
        });
        return;
      }

      // 80% thành công, 20% thất bại để mô phỏng thực tế
      const shouldSucceed = Math.random() > 0.2;

      if (shouldSucceed) {
        resolve({
          success: true,
          message: "Kết nối PACS thành công!",
          details: {
            host: config.host,
            port: config.port,
            localAE: config.localAETitle,
            remoteAE: config.remoteAETitle,
            responseTime: `${Math.floor(delay)}ms`,
            serverInfo: "PACS Server v2.1.0",
          },
        });
      } else {
        reject({
          success: false,
          message:
            "Không thể kết nối đến PACS server. Vui lòng kiểm tra cấu hình mạng.",
          errorCode: "CONNECTION_TIMEOUT",
        });
      }
    }, delay);
  });
};

/**
 * Mock VNA Test Connection
 * @param {Object} config - Cấu hình VNA
 * @param {string} config.baseUrl - Base URL của VNA
 * @param {string} config.qidoEndpoint - QIDO endpoint
 * @param {string} config.wadoEndpoint - WADO endpoint
 * @param {string} config.stowEndpoint - STOW endpoint
 * @param {string} config.authType - Loại xác thực (none/basic/token)
 * @param {string} config.username - Username (nếu authType = basic)
 * @param {string} config.password - Password (nếu authType = basic)
 * @param {string} config.token - Token (nếu authType = token)
 * @returns {Promise} - Promise với kết quả test
 */
export const mockVnaTest = (config) => {
  return new Promise((resolve, reject) => {
    // Mô phỏng thời gian request: 1-2 giây
    const delay = 1000 + Math.random() * 1000;

    setTimeout(() => {
      // Validation baseUrl
      if (!config.baseUrl) {
        reject({
          success: false,
          message: "Base URL không được để trống",
        });
        return;
      }

      // Rule: baseUrl phải là URL hợp lệ
      try {
        new URL(config.baseUrl);
      } catch {
        reject({
          success: false,
          message:
            "Base URL không hợp lệ. Vui lòng nhập URL đầy đủ (VD: https://vna.example.com)",
        });
        return;
      }

      // Validation auth
      if (config.authType === "basic") {
        if (!config.username || !config.password) {
          reject({
            success: false,
            message:
              "Username và Password không được để trống khi sử dụng Basic Auth",
          });
          return;
        }
      }

      if (config.authType === "token") {
        if (!config.token) {
          reject({
            success: false,
            message: "Token không được để trống khi sử dụng Token Auth",
          });
          return;
        }
      }

      // 85% thành công, 15% thất bại
      const shouldSucceed = Math.random() > 0.15;

      if (shouldSucceed) {
        resolve({
          success: true,
          message: "Kết nối VNA thành công!",
          details: {
            baseUrl: config.baseUrl,
            qido: config.qidoEndpoint || "/qido-rs",
            wado: config.wadoEndpoint || "/wado-rs",
            stow: config.stowEndpoint || "/stow-rs",
            authType: config.authType,
            responseTime: `${Math.floor(delay)}ms`,
            serverVersion: "DICOMweb v3.2.1",
            capabilities: ["QIDO-RS", "WADO-RS", "STOW-RS"],
          },
        });
      } else {
        // Random error types
        const errors = [
          {
            message: "Không thể kết nối đến VNA server. Vui lòng kiểm tra URL.",
            errorCode: "CONNECTION_REFUSED",
          },
          {
            message:
              "Xác thực thất bại. Vui lòng kiểm tra username/password hoặc token.",
            errorCode: "AUTH_FAILED",
          },
          {
            message: "VNA server không hỗ trợ DICOMweb.",
            errorCode: "UNSUPPORTED_PROTOCOL",
          },
        ];

        const error = errors[Math.floor(Math.random() * errors.length)];
        reject({
          success: false,
          ...error,
        });
      }
    }, delay);
  });
};

/**
 * Mock Save PACS Configuration
 * @param {Object} config - Cấu hình PACS
 * @returns {Promise} - Promise với kết quả lưu
 */
export const mockSavePacsConfig = (config) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      // Lưu vào localStorage
      localStorage.setItem("pacsConfig", JSON.stringify(config));
      resolve({
        success: true,
        message: "Đã lưu cấu hình PACS thành công!",
      });
    }, 500);
  });
};

/**
 * Mock Save VNA Configuration
 * @param {Object} config - Cấu hình VNA
 * @returns {Promise} - Promise với kết quả lưu
 */
export const mockSaveVnaConfig = (config) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      // Lưu vào localStorage
      localStorage.setItem("vnaConfig", JSON.stringify(config));
      resolve({
        success: true,
        message: "Đã lưu cấu hình VNA thành công!",
      });
    }, 500);
  });
};

/**
 * Load PACS Configuration từ localStorage
 * @returns {Object|null} - Cấu hình PACS hoặc null
 */
export const loadPacsConfig = () => {
  try {
    const config = localStorage.getItem("pacsConfig");
    return config ? JSON.parse(config) : null;
  } catch {
    return null;
  }
};

/**
 * Load VNA Configuration từ localStorage
 * @returns {Object|null} - Cấu hình VNA hoặc null
 */
export const loadVnaConfig = () => {
  try {
    const config = localStorage.getItem("vnaConfig");
    return config ? JSON.parse(config) : null;
  } catch {
    return null;
  }
};
